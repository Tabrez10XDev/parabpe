#pragma once

#include <string>
#include <string_view>

#include "parabpe/types.hpp"

namespace parabpe {

// Performance-core count when the CPU has two performance levels (Apple Silicon), else -1.
int performance_cores();

// Thread count for phase 3: cfg.threads when merge_threads is 0; min(threads, p_cores) for
// kMergeThreadsAuto, or threads when p_cores is unknown (< 1); otherwise merge_threads.
int resolve_merge_threads(const TrainConfig& cfg, int p_cores = performance_cores());

// Phase 1 only (chunk + pre-tokenize). Fills t.chunk and t.pretok when t is given.
WordCounts count_words(std::string_view text, const TrainConfig& cfg, PhaseTimings* t = nullptr);

// Full pipeline on in-memory text: chunk, pre-tokenize, build the word table, pairs, merges.
// Phase 1 runs on cfg.threads, phase 3 on resolve_merge_threads(cfg). The result's vocab follows
// SPEC.md §6.
TrainResult train_text(std::string_view text, const TrainConfig& cfg);

// Same, on a corpus file (memory-mapped); the timings include opening the file.
TrainResult train(const std::string& input_path, const TrainConfig& cfg);

}  // namespace parabpe
