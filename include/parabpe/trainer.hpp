#pragma once

#include <string>
#include <string_view>

#include "parabpe/types.hpp"

namespace parabpe {

// Phase 1 only (chunk + pre-tokenize). Fills t.chunk and t.pretok when t is given.
WordCounts count_words(std::string_view text, const TrainConfig& cfg, PhaseTimings* t = nullptr);

// Full pipeline on in-memory text: chunk, pre-tokenize, build the word table, pairs, merges.
// Sets the thread count to cfg.threads. The result's vocab follows SPEC.md §6.
TrainResult train_text(std::string_view text, const TrainConfig& cfg);

// Same, on a corpus file (memory-mapped); the timings include opening the file.
TrainResult train(const std::string& input_path, const TrainConfig& cfg);

}  // namespace parabpe
