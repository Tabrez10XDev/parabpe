#include "trace.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace parabpe::detail {
namespace {

constexpr const char* kNames[kTraceKinds] = {"CHUNK",      "PRETOK_UNIT",  "SHARD_REDUCE",
                                             "SELECT",     "SERIAL_APPLY", "OWNER_MERGE",
                                             "ZERO_SWEEP", "ORDER_UPDATE"};
constexpr const char* kArgNames[kTraceKinds] = {"bytes",  "bytes", "shard", "merges",
                                                "merges", "owner", "owner", "changed_pairs"};
constexpr const char* kArg2Names[kTraceKinds] = {nullptr, "unit",  nullptr, nullptr,
                                                 nullptr, nullptr, nullptr, nullptr};
constexpr const char* kCategories[kTraceKinds] = {"phase1", "phase1", "reduce", "phase3",
                                                  "phase3", "phase3", "phase3", "phase3"};

// Which CPU numbers are efficiency cores. Apple numbers the E-cores first: on the M3 Pro, CPUs 0-5
// are E and 6-10 are P, both in ioreg's cluster-type and in pthread_cpu_number_np. That order is
// the assumption here; hw.perflevel1 gives the E-core count.
struct CoreMap {
  int e_cores = -1;  // CPUs below this are E-cores; -1 = unknown, no labels
  std::string rule;
};

CoreMap core_map() {
  CoreMap map;
#if defined(__APPLE__)
  int levels = 0;
  size_t size = sizeof(levels);
  if (sysctlbyname("hw.nperflevels", &levels, &size, nullptr, 0) != 0 || levels != 2) return map;
  char name[64] = {};
  size = sizeof(name) - 1;
  if (sysctlbyname("hw.perflevel1.name", name, &size, nullptr, 0) != 0) return map;
  int e = 0;
  size = sizeof(e);
  if (std::string(name) != "Efficiency" ||
      sysctlbyname("hw.perflevel1.logicalcpu", &e, &size, nullptr, 0) != 0) {
    return map;
  }
  map.e_cores = e;
  map.rule = "cpu < " + std::to_string(e) +
             " is an E-core (hw.perflevel1.logicalcpu; Apple numbers E-cores first)";
#endif
  return map;
}

}  // namespace

const char* trace_name(TraceKind kind) {
  return kNames[static_cast<int>(kind)];
}
const char* trace_arg_name(TraceKind kind) {
  return kArgNames[static_cast<int>(kind)];
}

const char* trace_arg2_name(TraceKind kind) {
  return kArg2Names[static_cast<int>(kind)];
}

Tracer::Tracer(int threads) : buffers_(threads > 0 ? threads : 1), origin_(omp_get_wtime()) {}

uint64_t Tracer::count(TraceKind kind) const {
  uint64_t n = 0;
  for (const Buffer& b : buffers_) {
    for (const TraceEvent& e : b.events) n += e.kind == kind;
  }
  return n;
}

void Tracer::write_json(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write trace file " + path);
  const CoreMap cores = core_map();
  size_t total = 0;
  for (const Buffer& b : buffers_) total += b.events.size();

  out << "{\"displayTimeUnit\": \"ms\",\n\"otherData\": {\"tool\": \"parabpe\", \"threads\": "
      << buffers_.size() << ", \"events\": " << total << ", \"dropped\": " << dropped()
      << ", \"cpu\": \"sampled when each event starts; macOS has no thread pinning\"";
  if (cores.e_cores >= 0) {
    out << ", \"e_cores\": " << cores.e_cores << ", \"core_rule\": \"" << cores.rule << "\"";
  }
  out << "},\n\"traceEvents\": [\n";
  out << "{\"name\": \"process_name\", \"ph\": \"M\", \"pid\": 1, \"tid\": 0, "
         "\"args\": {\"name\": \"parabpe train\"}}";
  for (size_t t = 0; t < buffers_.size(); ++t) {
    out << ",\n{\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 1, \"tid\": " << t
        << ", \"args\": {\"name\": \"omp thread " << t << "\"}}";
    out << ",\n{\"name\": \"thread_sort_index\", \"ph\": \"M\", \"pid\": 1, \"tid\": " << t
        << ", \"args\": {\"sort_index\": " << t << "}}";
  }
  char line[320];
  for (size_t t = 0; t < buffers_.size(); ++t) {
    for (const TraceEvent& e : buffers_[t].events) {
      const int k = static_cast<int>(e.kind);
      const double ts = (e.t0 - origin_) * 1e6, dur = (e.t1 - e.t0) * 1e6;
      int n = std::snprintf(line, sizeof(line),
                            ",\n{\"name\": \"%s\", \"cat\": \"%s\", \"ph\": \"X\", \"pid\": 1, "
                            "\"tid\": %zu, \"ts\": %.3f, \"dur\": %.3f, \"args\": {\"%s\": %llu, "
                            "\"cpu\": %u",
                            kNames[k], kCategories[k], t, ts, dur > 0 ? dur : 0.0, kArgNames[k],
                            static_cast<unsigned long long>(e.arg), e.cpu);
      out.write(line, n);
      if (kArg2Names[k]) {
        out << ", \"" << kArg2Names[k] << "\": " << static_cast<unsigned long long>(e.arg2);
      }
      if (cores.e_cores >= 0) {
        out << ", \"core\": \"" << (static_cast<int>(e.cpu) < cores.e_cores ? 'E' : 'P') << "\"";
      }
      out << "}}";
    }
  }
  out << "\n]}\n";
  if (!out) throw std::runtime_error("error writing trace file " + path);
}

TraceSession::TraceSession(const std::string& path, int threads)
    : path_(path), previous_(g_tracer) {
  if (path_.empty()) return;
  tracer_ = std::make_unique<Tracer>(threads);
  g_tracer = tracer_.get();
}

TraceSession::~TraceSession() {
  if (tracer_) g_tracer = previous_;
}

void TraceSession::finish() {
  if (!tracer_) return;
  g_tracer = previous_;  // stop recording before the file is written
  tracer_->write_json(path_);
}

}  // namespace parabpe::detail
