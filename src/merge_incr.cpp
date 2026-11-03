#include <stdexcept>

#include "strategies.hpp"

namespace parabpe::detail {
namespace {

// S3 (serial): inverted index + whole-word diff updates + lazy max-heap.
// S6 (parallel): persistent omp region, owner-computes word partition with private
// indexes, padded per-thread delta buffers, threshold on tokens touched.
class IncrEngine final : public MergeEngine {
 public:
  explicit IncrEngine(bool parallel) : parallel_(parallel) {}

  std::vector<Merge> run(WordTable& /*table*/, const TrainConfig& /*cfg*/,
                         PhaseTimings& /*t*/) override {
    throw std::logic_error(parallel_ ? "IncrPar: not implemented (S6)"
                                     : "IncrSerial: not implemented (S3)");
  }

 private:
  bool parallel_;
};

}  // namespace

std::unique_ptr<MergeEngine> make_incr_engine(bool parallel) {
  return std::make_unique<IncrEngine>(parallel);
}

}  // namespace parabpe::detail
