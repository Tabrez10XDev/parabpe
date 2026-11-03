#include <stdexcept>

#include "strategies.hpp"

namespace parabpe::detail {
namespace {

// S4: dynamic schedule over work units; per-thread match_data + local maps pre-bucketed
// by high hash bits; after a barrier, thread t merges bucket t from all local maps.
class OmpPreTokenizer final : public PreTokenizer {
 public:
  WordCounts run(const std::vector<WorkUnit>& /*units*/, const TrainConfig& /*cfg*/) override {
    throw std::logic_error("OmpPreTokenizer: not implemented (S4)");
  }
};

}  // namespace

std::unique_ptr<PreTokenizer> make_omp_pretokenizer() {
  return std::make_unique<OmpPreTokenizer>();
}

}  // namespace parabpe::detail
