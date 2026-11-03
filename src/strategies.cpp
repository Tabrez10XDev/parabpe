#include "strategies.hpp"

#include <stdexcept>

namespace parabpe {

std::unique_ptr<PreTokenizer> make_pretokenizer(PretokStrategy strategy) {
  switch (strategy) {
    case PretokStrategy::Serial: return detail::make_serial_pretokenizer();
    case PretokStrategy::OMP: return detail::make_omp_pretokenizer();
  }
  throw std::invalid_argument("unknown PretokStrategy");
}

std::unique_ptr<MergeEngine> make_merge_engine(MergeStrategy strategy) {
  switch (strategy) {
    case MergeStrategy::NaiveSerial: return detail::make_naive_engine(false);
    case MergeStrategy::NaivePar: return detail::make_naive_engine(true);
    case MergeStrategy::IncrSerial: return detail::make_incr_engine(false);
    case MergeStrategy::IncrPar: return detail::make_incr_engine(true);
  }
  throw std::invalid_argument("unknown MergeStrategy");
}

}  // namespace parabpe
