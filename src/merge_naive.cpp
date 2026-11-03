#include <omp.h>

#include <ankerl/unordered_dense.h>
#include <bit>
#include <cstdint>
#include <vector>

#include "bpe_common.hpp"
#include "strategies.hpp"

namespace parabpe::detail {
namespace {

using PairCounts = ankerl::unordered_dense::map<PairKey, uint64_t>;

// SPEC.md §3: every adjacent pair, overlapping ones included, weighted by word frequency.
PairCounts count_pairs(const WordTable& t) {
  PairCounts counts;
  for (size_t w = 0; w < t.size(); ++w) {
    const TokenId* tok = t.tokens.data() + t.offset[w];
    for (uint32_t i = 0; i + 1 < t.len[w]; ++i)
      counts[make_pair_key(tok[i], tok[i + 1])] += t.freq[w];
  }
  return counts;
}

// Best pair of `counts` by SPEC.md §4; false if there are no pairs.
bool best_of(const PairCounts& counts, const std::vector<std::string>& token_bytes, uint64_t& count,
             PairKey& pair) {
  bool found = false;
  for (const auto& [k, c] : counts) {
    if (!found || better_pair(c, k, count, pair, token_bytes)) {
      count = c;
      pair = k;
      found = true;
    }
  }
  return found;
}

// Per-thread and per-shard state is written in hot loops: keep each on its own cache line.
struct alignas(64) LocalPairs {
  std::vector<PairCounts> buckets;  // bucket b holds the pairs whose low hash bits are b
};
struct alignas(64) Shard {
  PairCounts counts;
  uint64_t best_count = 0;
  PairKey best = 0;
  bool found = false;
};

// S7: one full recount per merge, in parallel. Each thread counts its words (dynamic schedule)
// into local maps pre-bucketed by the low hash bits (ankerl indexes by the high bits; see S4).
// After the barrier, bucket b of every thread is reduced into shard b, which also finds its best
// pair; the master then takes the best of the shard winners. better_pair is a total order on
// distinct pairs, so the winner does not depend on the partition.
class ParallelCounter {
 public:
  explicit ParallelCounter(int threads) : threads_(threads), locals_(threads) {}

  bool best_pair(const WordTable& t, const std::vector<std::string>& token_bytes, PairKey& best) {
    const int64_t n_words = static_cast<int64_t>(t.size());
#pragma omp parallel num_threads(threads_)
    {
      const int n_threads = omp_get_num_threads();
      const unsigned n_buckets = std::bit_ceil(static_cast<unsigned>(n_threads));
      const uint64_t mask = n_buckets - 1;
      const ankerl::unordered_dense::hash<PairKey> hash;
      LocalPairs& mine = locals_[omp_get_thread_num()];
      mine.buckets.resize(n_buckets);
      for (PairCounts& b : mine.buckets) b.clear();

#pragma omp for schedule(dynamic, 1024)
      for (int64_t w = 0; w < n_words; ++w) {
        const TokenId* tok = t.tokens.data() + t.offset[w];
        for (uint32_t i = 0; i + 1 < t.len[w]; ++i) {
          const PairKey k = make_pair_key(tok[i], tok[i + 1]);
          mine.buckets[hash(k) & mask][k] += t.freq[w];
        }
      }  // implicit barrier: every local map is complete

#pragma omp single
      shards_.resize(n_buckets);  // implicit barrier
#pragma omp for schedule(dynamic, 1)
      for (unsigned b = 0; b < n_buckets; ++b) {
        Shard& s = shards_[b];
        s.counts.clear();
        for (int th = 0; th < n_threads; ++th) {
          for (const auto& [k, c] : locals_[th].buckets[b]) s.counts[k] += c;
        }
        s.found = best_of(s.counts, token_bytes, s.best_count, s.best);
      }
    }

    bool found = false;
    uint64_t best_count = 0;
    for (const Shard& s : shards_) {
      if (s.found && (!found || better_pair(s.best_count, s.best, best_count, best, token_bytes))) {
        best_count = s.best_count;
        best = s.best;
        found = true;
      }
    }
    return found;
  }

 private:
  int threads_;
  std::vector<LocalPairs> locals_;
  std::vector<Shard> shards_;
};

// S2 (serial) / S7 (parallel): recount all pairs every merge. Small configs only.
class NaiveEngine final : public MergeEngine {
 public:
  explicit NaiveEngine(bool parallel) : parallel_(parallel) {}

  std::vector<Merge> run(WordTable& table, const TrainConfig& cfg, PhaseTimings& t) override {
    const size_t n_merges = num_merges(cfg);
    std::vector<std::string> token_bytes = initial_token_bytes(cfg);
    std::vector<Merge> merges;
    merges.reserve(n_merges);
    ParallelCounter counter(cfg.threads);
    auto find_best = [&](PairKey& best) {
      if (parallel_) return counter.best_pair(table, token_bytes, best);
      uint64_t count = 0;
      return best_of(count_pairs(table), token_bytes, count, best);
    };

    // Phase 2 is the first full count; every later recount is part of the merge loop.
    const double t0 = omp_get_wtime();
    PairKey best = 0;
    bool found = n_merges > 0 && find_best(best);
    const double t1 = omp_get_wtime();
    t.pairs = t1 - t0;

    const auto n_words = static_cast<int64_t>(table.size());
    for (size_t i = 0; i < n_merges && found; ++i) {
      if (i > 0) found = find_best(best);
      if (!found) break;  // nothing left to merge
      const TokenId a = pair_first(best), b = pair_second(best);
      const TokenId z = static_cast<TokenId>(token_bytes.size());
      // Words are disjoint ranges of table.tokens, so they can be rewritten independently.
#pragma omp parallel for schedule(dynamic, 1024) num_threads(cfg.threads) if (parallel_)
      for (int64_t w = 0; w < n_words; ++w) {
        table.len[w] = apply_merge(table.tokens.data() + table.offset[w], table.len[w], a, b, z);
      }
      merges.push_back({a, b, z});
      token_bytes.push_back(token_bytes[a] + token_bytes[b]);
    }
    t.merge = omp_get_wtime() - t1;
    return merges;
  }

 private:
  bool parallel_;
};

}  // namespace

std::unique_ptr<MergeEngine> make_naive_engine(bool parallel) {
  return std::make_unique<NaiveEngine>(parallel);
}

}  // namespace parabpe::detail
