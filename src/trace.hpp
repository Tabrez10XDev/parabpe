#pragma once

// B6: `--trace FILE` writes Chrome Trace Event JSON (open it in ui.perfetto.dev or
// chrome://tracing). Off by default, and then each event site costs one load of a global pointer
// and a branch. On, every thread appends to its own cache-line-aligned buffer, and nothing is
// shared until the run ends and the file is written.
//
// Each event carries the CPU its thread was on when the event started. macOS has no thread
// pinning, so this is a sample: a thread may move between P- and E-cores during an event.

#include <omp.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sched.h>
#endif

namespace parabpe::detail {

enum class TraceKind : uint8_t {
  Chunk,        // chunking: the whole step on the master, and each thread's special-token scan
  PretokUnit,   // one work unit pre-tokenized; arg = bytes, arg2 = the unit's index (B10)
  ShardReduce,  // one shard reduced (phase 1 word maps or a parallel step's pair deltas)
  Select,       // incr-batch: one round's selection on the master; arg = merges in the round
  SerialApply,  // incr-batch: a round applied by the master alone; arg = merges in the round
  OwnerMerge,   // one owner's words rewritten in a parallel step; arg = owner
  ZeroSweep,    // pairs at count 0 erased from one owner's index; arg = owner
  OrderUpdate,  // incr-batch: pair counts and heap updated after a round; arg = changed pairs
};
inline constexpr int kTraceKinds = 8;

const char* trace_name(TraceKind kind);      // "CHUNK", "PRETOK_UNIT", ...
const char* trace_arg_name(TraceKind kind);  // what `arg` counts: "bytes", "owner", ...
const char* trace_arg2_name(TraceKind kind);  // what `arg2` holds, or nullptr when unused

struct TraceEvent {
  double t0, t1;  // omp_get_wtime()
  uint64_t arg;
  uint32_t cpu;  // sampled when the event started
  TraceKind kind;
  uint64_t arg2 = 0;  // see trace_arg2_name
};

// The CPU the calling thread runs on right now (0 where the OS can't tell).
inline uint32_t current_cpu() {
#if defined(__APPLE__)
  size_t cpu = 0;
  pthread_cpu_number_np(&cpu);
  return static_cast<uint32_t>(cpu);
#elif defined(__linux__)
  const int cpu = sched_getcpu();
  return cpu < 0 ? 0u : static_cast<uint32_t>(cpu);
#else
  return 0;
#endif
}

class Tracer {
 public:
  explicit Tracer(int threads);

  // Appends to the calling OpenMP thread's buffer. A thread number with no buffer is counted as
  // dropped (it would mean a parallel region wider than cfg.threads).
  void add(const TraceEvent& e) {
    const int t = omp_get_thread_num();
    if (t >= 0 && static_cast<size_t>(t) < buffers_.size()) {
      buffers_[t].events.push_back(e);
    } else {
      dropped_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  int threads() const { return static_cast<int>(buffers_.size()); }
  const std::vector<TraceEvent>& events(int thread) const { return buffers_[thread].events; }
  uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }
  uint64_t count(TraceKind kind) const;

  // Chrome Trace Event JSON; timestamps are microseconds since the tracer was created.
  void write_json(const std::string& path) const;

 private:
  struct alignas(64) Buffer {
    std::vector<TraceEvent> events;
  };
  std::vector<Buffer> buffers_;
  double origin_;
  std::atomic<uint64_t> dropped_{0};
};

// The active tracer, or nullptr (tracing off). Only changed outside parallel regions.
inline Tracer* g_tracer = nullptr;

// Installs a tracer for one training run when `path` is non-empty, and always uninstalls it,
// also when the run throws. finish() writes the file.
class TraceSession {
 public:
  TraceSession(const std::string& path, int threads);
  ~TraceSession();
  TraceSession(const TraceSession&) = delete;
  TraceSession& operator=(const TraceSession&) = delete;
  void finish();

 private:
  std::string path_;
  std::unique_ptr<Tracer> tracer_;
  Tracer* previous_ = nullptr;
};

// Records [construction, end()) as one event on the calling thread; end() runs at scope exit.
class TraceSpan {
 public:
  explicit TraceSpan(TraceKind kind, uint64_t arg = 0) : tracer_(g_tracer), arg_(arg), kind_(kind) {
    if (tracer_) {
      cpu_ = current_cpu();
      t0_ = omp_get_wtime();
    }
  }
  ~TraceSpan() { end(); }
  TraceSpan(const TraceSpan&) = delete;
  TraceSpan& operator=(const TraceSpan&) = delete;

  bool on() const { return tracer_ != nullptr; }
  void set_arg(uint64_t arg) { arg_ = arg; }
  void set_arg2(uint64_t arg2) { arg2_ = arg2; }
  void end() {
    if (!tracer_) return;
    tracer_->add({t0_, omp_get_wtime(), arg_, cpu_, kind_, arg2_});
    tracer_ = nullptr;
  }

 private:
  Tracer* tracer_;
  double t0_ = 0;
  uint64_t arg_;
  uint64_t arg2_ = 0;
  uint32_t cpu_ = 0;
  TraceKind kind_;
};

// Records [t0, now) for a section the caller already timed.
inline void trace_since(TraceKind kind, double t0, uint64_t arg) {
  if (Tracer* t = g_tracer) t->add({t0, omp_get_wtime(), arg, current_cpu(), kind});
}

}  // namespace parabpe::detail
