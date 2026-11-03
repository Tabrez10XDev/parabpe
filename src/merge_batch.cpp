#include <stdexcept>

#include "strategies.hpp"

namespace parabpe::detail {
namespace {

// TODO(B3): batched merge rounds (docs/notes/batched_merging.md).
class BatchEngine final : public MergeEngine {
 public:
  std::vector<Merge> run(WordTable&, const TrainConfig&, PhaseTimings&) override {
    throw std::logic_error("IncrBatch: not implemented (B3)");
  }
};

}  // namespace

std::unique_ptr<MergeEngine> make_batch_engine() {
  return std::make_unique<BatchEngine>();
}

}  // namespace parabpe::detail
