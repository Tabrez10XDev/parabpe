#include <omp.h>

#include <ankerl/unordered_dense.h>
#include <stdexcept>

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

// S2 (serial) / S7 (parallel): recount all pairs every merge. Small configs only.
class NaiveEngine final : public MergeEngine {
 public:
  explicit NaiveEngine(bool parallel) : parallel_(parallel) {}

  std::vector<Merge> run(WordTable& table, const TrainConfig& cfg, PhaseTimings& t) override {
    // TODO(S7): parallel path (omp for over words, thread-local maps, bucketed reduction).
    if (parallel_) throw std::logic_error("NaivePar: not implemented (S7)");

    const size_t n_merges = num_merges(cfg);
    std::vector<std::string> token_bytes = initial_token_bytes(cfg);
    std::vector<Merge> merges;
    merges.reserve(n_merges);

    // Phase 2 is the first full count; every later recount is part of the merge loop.
    const double t0 = omp_get_wtime();
    PairCounts counts = count_pairs(table);
    const double t1 = omp_get_wtime();
    t.pairs = t1 - t0;

    for (size_t i = 0; i < n_merges; ++i) {
      if (i > 0) counts = count_pairs(table);
      if (counts.empty()) break;  // nothing left to merge
      auto best = counts.begin();
      for (auto it = counts.begin(); it != counts.end(); ++it) {
        if (better_pair(it->second, it->first, best->second, best->first, token_bytes)) best = it;
      }
      const TokenId a = pair_first(best->first), b = pair_second(best->first);
      const TokenId z = static_cast<TokenId>(token_bytes.size());
      for (size_t w = 0; w < table.size(); ++w) {
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
