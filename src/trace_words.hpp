#pragma once

// B10: `--trace-words FILE` writes a sample of the word table for ParaBPE Studio: the most frequent
// words of at least kTraceWordMinBytes bytes, with the owner that rewrites each in phase 3. Studio
// replays merges.txt on these words to show them fusing, and reads from the trace which thread ran
// that owner in each round. Written after the run's clock stops.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "parabpe/types.hpp"

namespace parabpe::detail {

inline constexpr size_t kTraceWordCount = 400;
inline constexpr size_t kTraceWordMinBytes = 4;

struct TraceWord {
  uint32_t w;      // index in the word table (= in WordCounts, sorted by bytes)
  uint64_t count;  // occurrences in the corpus
  int64_t owner;   // IncrBatch owner in phase 3; -1 for the other engines
};

// The `n` most frequent words of at least kTraceWordMinBytes bytes, most frequent first (ties: by
// index). merge_cfg is the phase-3 config (threads = the merge team).
std::vector<TraceWord> trace_word_sample(const WordCounts& counts, MergeStrategy merge,
                                         const TrainConfig& merge_cfg, size_t n = kTraceWordCount);

// JSON, one word per line: {"w": 12, "count": 5, "owner": 1, "hex": "746865"}.
void write_trace_words(const std::string& path, const WordCounts& counts, MergeStrategy merge,
                       const TrainConfig& merge_cfg);

}  // namespace parabpe::detail
