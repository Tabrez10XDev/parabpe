#pragma once

#include <memory>
#include <vector>

#include "parabpe/corpus.hpp"
#include "parabpe/types.hpp"

namespace parabpe {

// Phase 1: regex pre-tokenization + word counting.
class PreTokenizer {
 public:
  virtual ~PreTokenizer() = default;
  virtual WordCounts run(const std::vector<WorkUnit>& units, const TrainConfig& cfg) = 0;
};

std::unique_ptr<PreTokenizer> make_pretokenizer(PretokStrategy strategy);

}  // namespace parabpe
