#pragma once

#include <memory>
#include <string>
#include <vector>

#include "parabpe/types.hpp"

namespace parabpe {

// Converts unique words into the SoA table of byte-token ids.
WordTable build_word_table(const WordCounts& counts);

// Phases 2 + 3: pair statistics and the merge loop. Merges are appended until
// cfg.vocab_size is reached or no pairs remain. Ties are broken per SPEC.md §4.
// The engine fills t.pairs (phase 2) and t.merge (phase 3); words in `table` are rewritten.
class MergeEngine {
 public:
  virtual ~MergeEngine() = default;
  virtual std::vector<Merge> run(WordTable& table, const TrainConfig& cfg, PhaseTimings& t) = 0;
};

std::unique_ptr<MergeEngine> make_merge_engine(MergeStrategy strategy);

}  // namespace parabpe
