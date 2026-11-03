#include <omp.h>

#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bpe_common.hpp"
#include "strategies.hpp"

namespace parabpe::detail {
namespace {

using PairCounts = ankerl::unordered_dense::map<PairKey, int64_t>;
using PairIndex = ankerl::unordered_dense::map<PairKey, std::vector<uint32_t>>;  // pair -> word ids

struct HeapEntry {
  uint64_t count;
  PairKey pair;
};

bool contains_pair(const TokenId* tok, uint32_t len, TokenId a, TokenId b) {
  for (uint32_t i = 0; i + 1 < len; ++i) {
    if (tok[i] == a && tok[i + 1] == b) return true;
  }
  return false;
}

// Appends w unless it is already the last entry (a word visits its pairs in order, so repeats of
// the same pair within one word are consecutive). Older duplicates are filtered by last_touched.
void index_word(std::vector<uint32_t>& ids, uint32_t w) {
  if (ids.empty() || ids.back() != w) ids.push_back(w);
}

// Global pair counts split into power-of-two shards by the low hash bits (ankerl maps index by
// the high bits, see S4), so the parallel reduction can fill shard b without locks.
class ShardedCounts {
 public:
  void reset(unsigned n_shards) {
    shards_.assign(n_shards, {});
    mask_ = n_shards - 1;
  }
  size_t shard_of(PairKey k) const { return mask_ ? hash_(k) & mask_ : 0; }  // serial: no hash
  PairCounts& shard(size_t s) { return shards_[s].counts; }
  size_t n_shards() const { return shards_.size(); }

  const int64_t* find(PairKey k) const {
    const PairCounts& m = shards_[shard_of(k)].counts;
    const auto it = m.find(k);
    return it == m.end() ? nullptr : &it->second;
  }
  int64_t add(PairKey k, int64_t d) { return shards_[shard_of(k)].counts[k] += d; }
  void erase(PairKey k) { shards_[shard_of(k)].counts.erase(k); }
  size_t size() const {
    size_t n = 0;
    for (const auto& s : shards_) n += s.counts.size();
    return n;
  }

 private:
  struct alignas(64) Shard {
    PairCounts counts;
  };
  std::vector<Shard> shards_;
  uint64_t mask_ = 0;
  ankerl::unordered_dense::hash<PairKey> hash_;
};

// A contiguous range of words and its private inverted index (S5). IncrSerial has one owner;
// IncrPar has one per thread, balanced by token count.
struct alignas(64) Owner {
  uint32_t begin = 0, end = 0;
  PairIndex index;
};

struct alignas(64) LocalCounts {
  std::vector<PairCounts> buckets;  // bucket b feeds shard b
};

// Splits [0, n_words) into n contiguous ranges with about the same number of tokens each.
std::vector<Owner> partition_words(const WordTable& t, size_t n) {
  std::vector<Owner> owners(n);
  const size_t total = t.tokens.size();
  uint32_t w = 0;
  size_t seen = 0;
  for (size_t o = 0; o < n; ++o) {
    owners[o].begin = w;
    const size_t target = total * (o + 1) / n;
    while (w < t.size() && (seen < target || o + 1 == n)) seen += t.len[w++];
    owners[o].end = w;
  }
  return owners;
}

// Phase 2, serial (S3): one owner, one shard.
void count_serial(const WordTable& t, Owner& owner, ShardedCounts& counts) {
  PairCounts& c = counts.shard(0);
  for (uint32_t w = owner.begin; w < owner.end; ++w) {
    const TokenId* tok = t.tokens.data() + t.offset[w];
    for (uint32_t i = 0; i + 1 < t.len[w]; ++i) {
      const PairKey k = make_pair_key(tok[i], tok[i + 1]);
      c[k] += static_cast<int64_t>(t.freq[w]);
      index_word(owner.index[k], w);
    }
  }
}

// Phase 2, parallel (S5): each owner builds its private index and counts into the running
// thread's local maps, pre-bucketed by shard; after the barrier, shard b sums bucket b of every
// thread. Every owner's index lists words in increasing order, like the serial index.
void count_parallel(const WordTable& t, std::vector<Owner>& owners, ShardedCounts& counts,
                    int threads) {
  std::vector<LocalCounts> locals(threads);
  const int n_owners = static_cast<int>(owners.size());
#pragma omp parallel num_threads(threads)
  {
    const int n_threads = omp_get_num_threads();
#pragma omp single
    counts.reset(std::bit_ceil(static_cast<unsigned>(n_threads)));  // implicit barrier
    LocalCounts& mine = locals[omp_get_thread_num()];
    mine.buckets.resize(counts.n_shards());

#pragma omp for schedule(dynamic, 1)
    for (int o = 0; o < n_owners; ++o) {
      Owner& owner = owners[o];
      for (uint32_t w = owner.begin; w < owner.end; ++w) {
        const TokenId* tok = t.tokens.data() + t.offset[w];
        for (uint32_t i = 0; i + 1 < t.len[w]; ++i) {
          const PairKey k = make_pair_key(tok[i], tok[i + 1]);
          mine.buckets[counts.shard_of(k)][k] += static_cast<int64_t>(t.freq[w]);
          index_word(owner.index[k], w);
        }
      }
    }  // implicit barrier: every local map is complete

    const int n_shards = static_cast<int>(counts.n_shards());
#pragma omp for schedule(dynamic, 1)
    for (int s = 0; s < n_shards; ++s) {
      PairCounts& shard = counts.shard(s);
      for (int th = 0; th < n_threads; ++th) {
        for (const auto& [k, c] : locals[th].buckets[s]) shard[k] += c;
      }
    }
  }
}

// S3 (serial) and S5 (parallel phase 2): inverted index + whole-word diff updates + lazy max-heap.
// IncrPar's phase 3 still runs serially over the owners' private indexes.
// TODO(S6): parallel phase 3 (persistent omp region, owners apply merges to their own words,
// padded per-thread delta buffers, bucketed delta reduce, threshold on tokens touched).
class IncrEngine final : public MergeEngine {
 public:
  explicit IncrEngine(bool parallel) : parallel_(parallel) {}

  std::vector<Merge> run(WordTable& table, const TrainConfig& cfg, PhaseTimings& t) override {
    const size_t n_merges = num_merges(cfg);
    std::vector<std::string> token_bytes = initial_token_bytes(cfg);
    std::vector<Merge> merges;
    merges.reserve(n_merges);

    // Heap order is SPEC.md §4. Token bytes never change once assigned, so stale entries keep a
    // stable position; they are dropped when popped (lazy deletion).
    auto heap_less = [&token_bytes](const HeapEntry& x, const HeapEntry& y) {
      return better_pair(y.count, y.pair, x.count, x.pair, token_bytes);
    };

    // Phase 2: counts, per-owner inverted indexes, heap.
    const double t0 = omp_get_wtime();
    std::vector<Owner> owners = partition_words(table, parallel_ ? cfg.threads : 1);
    ShardedCounts counts;
    if (parallel_) {
      count_parallel(table, owners, counts, cfg.threads);
    } else {
      counts.reset(1);
      count_serial(table, owners[0], counts);
    }
    std::vector<HeapEntry> heap;
    auto rebuild_heap = [&]() {
      heap.clear();
      heap.reserve(counts.size());
      for (size_t s = 0; s < counts.n_shards(); ++s) {
        for (const auto& [k, c] : counts.shard(s)) heap.push_back({static_cast<uint64_t>(c), k});
      }
      std::make_heap(heap.begin(), heap.end(), heap_less);
    };
    rebuild_heap();
    const double t1 = omp_get_wtime();
    t.pairs = t1 - t0;

    // Phase 3.
    PairCounts delta;
    for (uint32_t m = 0; m < n_merges; ++m) {
      // Pop until the top is live: its count is the current (positive) count of its pair.
      PairKey best = 0;
      bool found = false;
      while (!heap.empty()) {
        const HeapEntry top = heap.front();
        std::pop_heap(heap.begin(), heap.end(), heap_less);
        heap.pop_back();
        const int64_t* c = counts.find(top.pair);
        if (c && static_cast<uint64_t>(*c) == top.count) {
          best = top.pair;
          found = true;
          break;
        }
      }
      if (!found) break;  // nothing left to merge

      const TokenId a = pair_first(best), b = pair_second(best);
      const TokenId z = static_cast<TokenId>(token_bytes.size());
      // Before any push: the heap comparator reads the bytes of pairs that contain z.
      token_bytes.push_back(token_bytes[a] + token_bytes[b]);

      // Whole-word diff: subtract every pair of the old word, add every pair of the new one.
      // z is a fresh id, so the pairs new to a word are exactly the ones that contain z.
      delta.clear();
      for (Owner& owner : owners) {
        const auto idx = owner.index.find(best);
        if (idx == owner.index.end()) continue;
        // Take the word list out first: inserting new pairs below may move the map's storage.
        const std::vector<uint32_t> words = std::move(idx->second);
        owner.index.erase(idx);
        for (const uint32_t w : words) {
          if (table.last_touched[w] == m) continue;
          table.last_touched[w] = m;
          TokenId* tok = table.tokens.data() + table.offset[w];
          const uint32_t len = table.len[w];
          if (!contains_pair(tok, len, a, b)) continue;
          const auto f = static_cast<int64_t>(table.freq[w]);
          for (uint32_t i = 0; i + 1 < len; ++i) delta[make_pair_key(tok[i], tok[i + 1])] -= f;
          const uint32_t new_len = apply_merge(tok, len, a, b, z);
          table.len[w] = new_len;
          for (uint32_t i = 0; i + 1 < new_len; ++i) {
            const PairKey k = make_pair_key(tok[i], tok[i + 1]);
            delta[k] += f;
            if (tok[i] == z || tok[i + 1] == z) index_word(owner.index[k], w);
          }
        }
      }

      for (const auto& [k, d] : delta) {
        if (d == 0) continue;
        const int64_t c = counts.add(k, d);
        if (c > 0) {
          heap.push_back({static_cast<uint64_t>(c), k});
          std::push_heap(heap.begin(), heap.end(), heap_less);
        } else if (c == 0) {
          counts.erase(k);
          for (Owner& owner : owners) owner.index.erase(k);  // no word holds the pair any more
        } else {
          throw std::logic_error("IncrEngine: negative pair count");
        }
      }
      if (heap.size() > 4 * counts.size()) rebuild_heap();

      merges.push_back({a, b, z});
    }
    t.merge = omp_get_wtime() - t1;
    return merges;
  }

 private:
  bool parallel_;
};

}  // namespace

std::unique_ptr<MergeEngine> make_incr_engine(bool parallel) {
  return std::make_unique<IncrEngine>(parallel);
}

}  // namespace parabpe::detail
