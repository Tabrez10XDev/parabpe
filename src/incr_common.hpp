#pragma once

// Internal: the pieces of the incremental merge engines (src/merge_incr.cpp, src/merge_batch.cpp):
// sharded pair counts, owners with private inverted indexes, phase 2 counting, and one merge step.

#include <omp.h>

#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <bit>
#include <cstdint>
#include <utility>
#include <vector>

#include "bpe_common.hpp"
#include "trace.hpp"

namespace parabpe::detail {

using PairCounts = ankerl::unordered_dense::map<PairKey, int64_t>;
using PairIndex = ankerl::unordered_dense::map<PairKey, std::vector<uint32_t>>;  // pair -> word ids

struct HeapEntry {
  uint64_t count;
  PairKey pair;
};

inline bool contains_pair(const TokenId* tok, uint32_t len, TokenId a, TokenId b) {
  for (uint32_t i = 0; i + 1 < len; ++i) {
    if (tok[i] == a && tok[i + 1] == b) return true;
  }
  return false;
}

// Appends w unless it is already the last entry (a word visits its pairs in order, so repeats of
// the same pair within one word are consecutive). Older duplicates are filtered by last_touched.
inline void index_word(std::vector<uint32_t>& ids, uint32_t w) {
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

// A set of words and its private inverted index (S5). IncrSerial has one owner; IncrPar has one per
// thread, each a contiguous range balanced by token count unless --owner-map hashed says
// otherwise (B5); IncrBatch has hashed owners (B4), whose words are listed in increasing order
// instead of being a range.
struct alignas(64) Owner {
  uint32_t begin = 0, end = 0;
  std::vector<uint32_t> words;  // hashed owners only
  PairIndex index;

  template <class F>
  void for_each_word(F&& f) const {
    if (words.empty()) {
      for (uint32_t w = begin; w < end; ++w) f(w);
    } else {
      for (const uint32_t w : words) f(w);
    }
  }
};

struct alignas(64) LocalCounts {
  std::vector<PairCounts> buckets;  // bucket b feeds shard b
};

// Splits [0, n_words) into n contiguous ranges with about the same number of tokens each.
inline std::vector<Owner> partition_words(const WordTable& t, size_t n) {
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

inline uint64_t mix64(uint64_t x) {  // splitmix64 finalizer
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Splits the words into n owners by a hash of their block of 16 neighbours. The word table is
// sorted by bytes, so contiguous owners cluster the words of a pair in one owner (B0 measured
// busiest/mean 2.30 at 5 threads against 1.08 here); a whole block stays together for locality.
inline std::vector<Owner> partition_words_hashed(const WordTable& t, size_t n) {
  std::vector<Owner> owners(n);
  for (uint32_t w = 0; w < t.size(); ++w) owners[mix64(w >> 4) % n].words.push_back(w);
  return owners;
}

// Phase 2, serial (S3): one owner, one shard.
inline void count_serial(const WordTable& t, Owner& owner, ShardedCounts& counts) {
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
inline void count_parallel(const WordTable& t, std::vector<Owner>& owners, ShardedCounts& counts,
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
      owner.for_each_word([&](uint32_t w) {
        const TokenId* tok = t.tokens.data() + t.offset[w];
        for (uint32_t i = 0; i + 1 < t.len[w]; ++i) {
          const PairKey k = make_pair_key(tok[i], tok[i + 1]);
          mine.buckets[counts.shard_of(k)][k] += static_cast<int64_t>(t.freq[w]);
          index_word(owner.index[k], w);
        }
      });
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
inline size_t tokens_touched(const WordTable& table, const std::vector<Owner>& owners, PairKey pair,
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

// A step at least this big counts towards the owner-load statistics (B5).
inline constexpr uint64_t kBigStepTokens = 4096;

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

// One parallel step (S6). merge(table, owner, add) rewrites one owner's words and reports its
// pair-count changes through add(pair, delta). With erase_zero_pairs off, pairs that reached 0
// stay in the owners' indexes and the caller takes them from shard_delta[s].zero. Owners only touch
// their own words and index; shard s of the counts is only written by the thread reducing shard s.
// Deltas are summed over all threads before they are applied, so a count never passes through a
// transient 0 or negative value.
template <class MergeFn>
void parallel_step(WordTable& table, std::vector<Owner>& owners, ShardedCounts& counts,
                   MergeFn&& merge, std::vector<ThreadDelta>& thread_delta,
                   std::vector<ShardDelta>& shard_delta, int threads,
                   bool erase_zero_pairs = true) {
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
      const TraceSpan span(TraceKind::OwnerMerge, static_cast<uint64_t>(o));
      merge(table, owners[o],
            [&mine, &counts](PairKey k, int64_t d) { mine.buckets[counts.shard_of(k)][k] += d; });
    }  // implicit barrier: every delta is buffered

#pragma omp for schedule(dynamic, 1)
    for (int s = 0; s < n_shards; ++s) {
      const TraceSpan span(TraceKind::ShardReduce, static_cast<uint64_t>(s));
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

    if (erase_zero_pairs) {
#pragma omp for schedule(dynamic, 1)
      for (int o = 0; o < n_owners; ++o) {
        const TraceSpan span(TraceKind::ZeroSweep, static_cast<uint64_t>(o));
        for (const ShardDelta& sd : shard_delta) {
          for (const PairKey k : sd.zero) owners[o].index.erase(k);
        }
      }
    }
  }
}

}  // namespace parabpe::detail
