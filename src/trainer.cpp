#include "parabpe/trainer.hpp"

#include <omp.h>

#include "parabpe/corpus.hpp"
#include "parabpe/io.hpp"
#include "parabpe/merge_engine.hpp"
#include "parabpe/pretokenizer.hpp"

namespace parabpe {
namespace {

// Wall-clock lap timer over the phases (SPEC.md §6: omp_get_wtime()).
class Laps {
 public:
  explicit Laps(double start) : mark_(start) {}
  void lap(double& slot) {
    const double now = omp_get_wtime();
    slot = now - mark_;
    mark_ = now;
  }

 private:
  double mark_;
};

TrainResult run_pipeline(std::string_view text, const TrainConfig& cfg, double t0,
                         double chunk_extra) {
  omp_set_num_threads(cfg.threads);
  TrainResult r;
  WordCounts counts = count_words(text, cfg, &r.timings);
  r.timings.chunk += chunk_extra;

  Laps laps(omp_get_wtime());
  WordTable table = build_word_table(counts);
  laps.lap(r.timings.build);

  const auto engine = make_merge_engine(cfg.merge);
  r.merges = engine->run(table, cfg, r.timings);
  r.stats = engine->stats();
  r.vocab = build_vocab(r.merges, cfg.special_tokens);
  r.timings.total = omp_get_wtime() - t0;
  return r;
}

}  // namespace

WordCounts count_words(std::string_view text, const TrainConfig& cfg, PhaseTimings* t) {
  omp_set_num_threads(cfg.threads);
  PhaseTimings local;
  PhaseTimings& out = t ? *t : local;
  Laps laps(omp_get_wtime());
  // The OMP pipeline also chunks in parallel (S10); --pretok serial keeps the serial chunker.
  const int chunk_threads = cfg.pretok == PretokStrategy::OMP ? cfg.threads : 1;
  const auto units = make_work_units(text, cfg.special_tokens, cfg.work_unit_bytes, chunk_threads);
  laps.lap(out.chunk);
  WordCounts counts = make_pretokenizer(cfg.pretok)->run(units, cfg);
  laps.lap(out.pretok);
  return counts;
}

TrainResult train_text(std::string_view text, const TrainConfig& cfg) {
  return run_pipeline(text, cfg, omp_get_wtime(), 0.0);
}

TrainResult train(const std::string& input_path, const TrainConfig& cfg) {
  const double t0 = omp_get_wtime();
  const Corpus corpus = Corpus::open(input_path);
  return run_pipeline(corpus.text(), cfg, t0, omp_get_wtime() - t0);  // mmap counts as chunking
}

}  // namespace parabpe
