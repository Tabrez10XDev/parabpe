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

struct MergeStep {
  uint32_t m;  // merge number, used as the last_touched stamp
  PairKey pair;
  TokenId a, b, z;
};

// Rewrites the owner's words that contain step.pair and reports the pair-count changes through
// add(pair, delta). Whole-word diff: subtract every pair of the old word, add every pair of the new
// one. z is a fresh id, so the pairs new to a word are exactly the ones that contain z; only those
// are appended to the owner's index. Touches nothing outside this owner (its words, its index).
template <class Add>
void merge_owner(WordTable& table, Owner& owner, const MergeStep& s, Add&& add) {
  const auto idx = owner.index.find(s.pair);
  if (idx == owner.index.end()) return;
  // Take the word list out first: inserting new pairs below may move the map's storage.
  const std::vector<uint32_t> words = std::move(idx->second);
  owner.index.erase(idx);
  for (const uint32_t w : words) {
    if (table.last_touched[w] == s.m) continue;
    table.last_touched[w] = s.m;
    TokenId* tok = table.tokens.data() + table.offset[w];
    const uint32_t len = table.len[w];
    if (!contains_pair(tok, len, s.a, s.b)) continue;
    const auto f = static_cast<int64_t>(table.freq[w]);
    for (uint32_t i = 0; i + 1 < len; ++i) add(make_pair_key(tok[i], tok[i + 1]), -f);
    const uint32_t new_len = apply_merge(tok, len, s.a, s.b, s.z);
    table.len[w] = new_len;
    for (uint32_t i = 0; i + 1 < new_len; ++i) {
      const PairKey k = make_pair_key(tok[i], tok[i + 1]);
      add(k, f);
      if (tok[i] == s.z || tok[i + 1] == s.z) index_word(owner.index[k], w);
    }
  }
}

// Upper bound on the tokens a merge step rewrites, cut off at `limit`: every listed word has at
// least one token, so `limit` listed words already reach it without reading their lengths.
size_t tokens_touched(const WordTable& table, const std::vector<Owner>& owners, PairKey pair,
                      size_t limit) {
  size_t words = 0;
  for (const Owner& o : owners) {
    const auto it = o.index.find(pair);
    if (it != o.index.end()) words += it->second.size();
  }
  if (words >= limit) return words;
  size_t tokens = 0;
  for (const Owner& o : owners) {
    const auto it = o.index.find(pair);
    if (it == o.index.end()) continue;
    for (const uint32_t w : it->second) tokens += table.len[w];
  }
  return tokens;
}

// Per-thread delta buffer (S6), pre-bucketed by counts shard; written in the hot loop, so padded.
struct alignas(64) ThreadDelta {
  std::vector<PairCounts> buckets;
};
// Per-shard output of the delta reduction.
struct alignas(64) ShardDelta {
  PairCounts sum;               // deltas of this shard summed over threads
  std::vector<HeapEntry> push;  // pairs whose count changed and is still positive
  std::vector<PairKey> zero;    // pairs whose count reached 0
  bool negative = false;
};

// S3 (serial), S5 (parallel phase 2) and S6 (parallel phase 3): inverted index + whole-word diff
// updates + lazy max-heap. In IncrPar the master runs the merge loop; a step that touches at least
// cfg.par_threshold_tokens tokens opens a parallel region in which each owner merges its own words
// into padded per-thread delta buffers, shard s then sums and applies its deltas, and owners drop
// pairs that reached 0. Smaller steps run on the master alone, like IncrSerial. (A persistent
// region would cost a barrier on every one of the ~30k merges, most of which touch a handful of
// words.)
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

    auto push = [&](uint64_t c, PairKey k) {
      heap.push_back({c, k});
      std::push_heap(heap.begin(), heap.end(), heap_less);
    };

    // Phase 3.
    PairCounts delta;  // serial steps
    std::vector<ThreadDelta> thread_delta(parallel_ ? cfg.threads : 0);
    std::vector<ShardDelta> shard_delta(counts.n_shards());
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

      const MergeStep step{m, best, pair_first(best), pair_second(best),
                           static_cast<TokenId>(token_bytes.size())};
      // Before any push: the heap comparator reads the bytes of pairs that contain z.
      token_bytes.push_back(token_bytes[step.a] + token_bytes[step.b]);

      const bool par_step =
          parallel_ && cfg.threads > 1 &&
          tokens_touched(table, owners, best, cfg.par_threshold_tokens) >= cfg.par_threshold_tokens;
      if (par_step) {
        parallel_step(table, owners, counts, step, thread_delta, shard_delta, cfg.threads);
        for (const ShardDelta& sd : shard_delta) {
          if (sd.negative) throw std::logic_error("IncrEngine: negative pair count");
          for (const HeapEntry& e : sd.push) push(e.count, e.pair);
        }
      } else {
        delta.clear();
        for (Owner& owner : owners) {
          merge_owner(table, owner, step, [&delta](PairKey k, int64_t d) { delta[k] += d; });
        }
        for (const auto& [k, d] : delta) {
          if (d == 0) continue;
          const int64_t c = counts.add(k, d);
          if (c > 0) {
            push(static_cast<uint64_t>(c), k);
          } else if (c == 0) {
            counts.erase(k);
            for (Owner& owner : owners) owner.index.erase(k);  // no word holds the pair any more
          } else {
            throw std::logic_error("IncrEngine: negative pair count");
          }
        }
      }
      if (heap.size() > 4 * counts.size()) rebuild_heap();

      merges.push_back({step.a, step.b, step.z});
    }
    t.merge = omp_get_wtime() - t1;
    return merges;
  }

 private:
  // One parallel merge step (S6). Owners only touch their own words and index; shard s of the
  // counts is only written by the thread reducing shard s. Deltas are summed over all threads
  // before they are applied, so a count never passes through a transient 0 or negative value.
  static void parallel_step(WordTable& table, std::vector<Owner>& owners, ShardedCounts& counts,
                            const MergeStep& step, std::vector<ThreadDelta>& thread_delta,
                            std::vector<ShardDelta>& shard_delta, int threads) {
    const int n_owners = static_cast<int>(owners.size());
    const int n_shards = static_cast<int>(counts.n_shards());
#pragma omp parallel num_threads(threads)
    {
      const int n_threads = omp_get_num_threads();
      ThreadDelta& mine = thread_delta[omp_get_thread_num()];
      mine.buckets.resize(n_shards);
      for (PairCounts& b : mine.buckets) b.clear();

#pragma omp for schedule(dynamic, 1)
      for (int o = 0; o < n_owners; ++o) {
        merge_owner(table, owners[o], step, [&mine, &counts](PairKey k, int64_t d) {
          mine.buckets[counts.shard_of(k)][k] += d;
        });
      }  // implicit barrier: every delta is buffered

#pragma omp for schedule(dynamic, 1)
      for (int s = 0; s < n_shards; ++s) {
        ShardDelta& sd = shard_delta[s];
        sd.sum.clear();
        sd.push.clear();
        sd.zero.clear();
        sd.negative = false;
        for (int th = 0; th < n_threads; ++th) {
          for (const auto& [k, d] : thread_delta[th].buckets[s]) sd.sum[k] += d;
        }
        PairCounts& shard = counts.shard(s);
        for (const auto& [k, d] : sd.sum) {
          if (d == 0) continue;
          const int64_t c = (shard[k] += d);
          if (c > 0) {
            sd.push.push_back({static_cast<uint64_t>(c), k});
          } else if (c == 0) {
            shard.erase(k);
            sd.zero.push_back(k);
          } else {
            sd.negative = true;  // reported by the master: no exceptions out of the region
          }
        }
      }  // implicit barrier

#pragma omp for schedule(dynamic, 1)
      for (int o = 0; o < n_owners; ++o) {
        for (const ShardDelta& sd : shard_delta) {
          for (const PairKey k : sd.zero) owners[o].index.erase(k);
        }
      }
    }
  }

  bool parallel_;
};

}  // namespace

std::unique_ptr<MergeEngine> make_incr_engine(bool parallel) {
  return std::make_unique<IncrEngine>(parallel);
}

}  // namespace parabpe::detail
