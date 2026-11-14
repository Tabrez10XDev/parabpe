#include "parabpe/trainer.hpp"

#include <omp.h>

#include <algorithm>
#include <stdexcept>

#include "parabpe/corpus.hpp"
#include "parabpe/io.hpp"
#include "parabpe/merge_engine.hpp"
#include "parabpe/pretokenizer.hpp"
#include "trace.hpp"
#include "trace_words.hpp"

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

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
  // Phase 3 may use a different team (B8): every engine sizes its team, owners and parallel
  // rounds from threads, so it gets a copy with merge_threads in its place.
  TrainConfig merge_cfg = cfg;
  merge_cfg.threads = resolve_merge_threads(cfg);
  detail::TraceSession trace(cfg.trace_path, std::max(cfg.threads, merge_cfg.threads));
  TrainResult r;
  WordCounts counts = count_words(text, cfg, &r.timings);
  r.timings.chunk += chunk_extra;

  Laps laps(omp_get_wtime());
  WordTable table = build_word_table(counts);
  laps.lap(r.timings.build);

  omp_set_num_threads(merge_cfg.threads);
  const auto engine = make_merge_engine(cfg.merge);
  r.merges = engine->run(table, merge_cfg, r.timings);
  r.stats = engine->stats();
  r.vocab = build_vocab(r.merges, cfg.special_tokens);
  r.timings.total = omp_get_wtime() - t0;
  trace.finish();  // after the clock stops: writing the file is not part of the run
  if (!cfg.trace_words_path.empty()) {
    detail::write_trace_words(cfg.trace_words_path, counts, cfg.merge, merge_cfg);
  }
  return r;
}

}  // namespace

int performance_cores() {
#ifdef __APPLE__
  auto get = [](const char* name) {
    int value = 0;
    size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 ? value : -1;
  };
  if (get("hw.nperflevels") >= 2) return get("hw.perflevel0.physicalcpu");
#endif
  return -1;
}

int resolve_merge_threads(const TrainConfig& cfg, int p_cores) {
  if (cfg.merge_threads == 0) return cfg.threads;
  if (cfg.merge_threads == kMergeThreadsAuto) {
    return p_cores >= 1 ? std::min(cfg.threads, p_cores) : cfg.threads;
  }
  if (cfg.merge_threads < 1) throw std::invalid_argument("merge_threads must be >= 1, 0 or auto");
  return cfg.merge_threads;
}

WordCounts count_words(std::string_view text, const TrainConfig& cfg, PhaseTimings* t) {
  omp_set_num_threads(cfg.threads);
  PhaseTimings local;
  PhaseTimings& out = t ? *t : local;
  Laps laps(omp_get_wtime());
  // The OMP pipeline also chunks in parallel (S10); --pretok serial keeps the serial chunker.
  const int chunk_threads = cfg.pretok == PretokStrategy::OMP ? cfg.threads : 1;
  detail::TraceSpan chunk(detail::TraceKind::Chunk, text.size());
  const auto units = make_work_units(text, cfg.special_tokens, cfg.work_unit_bytes, chunk_threads);
  chunk.end();
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
