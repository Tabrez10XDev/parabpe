#pragma once

// Internal: constructors for each strategy, one per source file.

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "parabpe/merge_engine.hpp"
#include "parabpe/pretokenizer.hpp"

namespace parabpe::detail {

// Unique words -> WordCounts sorted by raw bytes (SPEC.md §6). Shared by the pre-tokenizers.
WordCounts sorted_word_counts(std::vector<std::pair<std::string_view, uint64_t>> entries);

std::unique_ptr<PreTokenizer> make_serial_pretokenizer();
std::unique_ptr<PreTokenizer> make_omp_pretokenizer();

std::unique_ptr<MergeEngine> make_naive_engine(bool parallel);
std::unique_ptr<MergeEngine> make_incr_engine(bool parallel);
std::unique_ptr<MergeEngine> make_batch_engine();

}  // namespace parabpe::detail
