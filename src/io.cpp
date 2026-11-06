#include "parabpe/io.hpp"

#include <sys/utsname.h>
#include <unistd.h>

#include <ankerl/unordered_dense.h>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace parabpe {
namespace {

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

std::string format_double(double x) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6f", x);
  return buf;
}

// ---- hardware info for timings.json ---------------------------------------------------------

struct Hardware {
  std::string cpu_model = "unknown", os = "unknown";
  int physical_cores = 0, logical_cores = 0;
  int performance_cores = -1, efficiency_cores = -1;  // Apple Silicon only (-1 = not reported)
};

#ifdef __APPLE__
std::string sysctl_string(const char* name) {
  size_t size = 0;
  if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) return "unknown";
  std::string s(size, '\0');
  if (sysctlbyname(name, s.data(), &size, nullptr, 0) != 0) return "unknown";
  s.resize(s.find('\0') == std::string::npos ? size : s.find('\0'));
  return s;
}

int sysctl_int(const char* name) {
  int value = 0;
  size_t size = sizeof(value);
  return sysctlbyname(name, &value, &size, nullptr, 0) == 0 ? value : -1;
}

Hardware detect_hardware() {
  Hardware h;
  h.cpu_model = sysctl_string("machdep.cpu.brand_string");
  h.physical_cores = sysctl_int("hw.physicalcpu");
  h.logical_cores = sysctl_int("hw.logicalcpu");
  h.performance_cores = sysctl_int("hw.perflevel0.physicalcpu");
  h.efficiency_cores = sysctl_int("hw.perflevel1.physicalcpu");
  h.os = "macOS " + sysctl_string("kern.osproductversion");
  return h;
}
#else
Hardware detect_hardware() {
  Hardware h;
  h.logical_cores = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
  // Physical cores = distinct (physical id, core id) pairs in /proc/cpuinfo.
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::set<std::pair<std::string, std::string>> cores;
  std::string line, physical_id = "0";
  while (std::getline(cpuinfo, line)) {
    const size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string key = line.substr(0, colon);
    while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
    const std::string value = colon + 2 <= line.size() ? line.substr(colon + 2) : "";
    if (key == "model name" && h.cpu_model == "unknown") h.cpu_model = value;
    if (key == "physical id") physical_id = value;
    if (key == "core id") cores.emplace(physical_id, value);
  }
  h.physical_cores = cores.empty() ? h.logical_cores : static_cast<int>(cores.size());
  utsname u{};
  if (uname(&u) == 0) h.os = std::string(u.sysname) + " " + u.release;
  return h;
}
#endif

// SHA of the source tree's HEAD, or "unknown" (no git, no commits, not a repo).
std::string git_sha() {
#ifdef PARABPE_SOURCE_DIR
  const std::string cmd = "git -C '" PARABPE_SOURCE_DIR "' rev-parse HEAD 2>/dev/null";
  if (FILE* p = popen(cmd.c_str(), "r")) {
    char buf[64] = {};
    const bool ok = std::fgets(buf, sizeof(buf), p) != nullptr;
    const int status = pclose(p);
    std::string sha = ok ? std::string(buf) : "";
    while (!sha.empty() && (sha.back() == '\n' || sha.back() == '\r')) sha.pop_back();
    if (status == 0 && sha.size() == 40) return sha;
  }
#endif
  return "unknown";
}

const char* name_of(PretokStrategy s) {
  return s == PretokStrategy::Serial ? "serial" : "omp";
}

const char* name_of(PretokReduce r) {
  return r == PretokReduce::Bucketed ? "bucketed" : "critical";
}

const char* name_of(OwnerMap m) {
  return m == OwnerMap::Hashed ? "hashed" : "contiguous";
}

const char* name_of(MergeStrategy s) {
  switch (s) {
    case MergeStrategy::NaiveSerial: return "naive-serial";
    case MergeStrategy::IncrSerial: return "incr-serial";
    case MergeStrategy::NaivePar: return "naive-par";
    case MergeStrategy::IncrPar: return "incr-par";
    case MergeStrategy::IncrBatch: return "incr-batch";
  }
  return "unknown";
}

// The additive "batch" block of timings.json, written for incr-batch only.
std::string format_batch_stats(const MergeStats& s, const TrainConfig& cfg) {
  static constexpr std::array<const char*, 9> kSizes = {"1",     "2",     "3-4",    "5-8", "9-16",
                                                        "17-32", "33-64", "65-128", "129+"};
  static constexpr std::array<const char*, 5> kStops = {"disjoint", "tie", "self", "budget",
                                                        "capped"};
  std::ostringstream o;
  o << "  \"batch\": {\"rounds\": " << s.rounds << ", \"max_batch\": " << s.max_batch
    << ", \"parallel_rounds\": " << s.parallel_rounds << ", \"changed_pairs\": " << s.changed_pairs
    << ", \"small_rounds\": " << s.small_rounds
    << ", \"select\": " << format_double(s.select_s) << ", \"order\": " << format_double(s.order_s)
    << ", \"owners_per_thread\": " << cfg.owners_per_thread
    << ", \"scan_cap\": " << cfg.batch_scan_cap << ", \"verified\": "
    << (cfg.verify_batches ? "true" : "false") << ", \"sizes\": {";
  for (size_t i = 0; i < kSizes.size(); ++i) {
    o << (i ? ", " : "") << '"' << kSizes[i] << "\": " << s.batch_hist[i];
  }
  o << "}, \"stops\": {";
  for (size_t i = 0; i < kStops.size(); ++i) {
    o << (i ? ", " : "") << '"' << kStops[i] << "\": " << s.stops[i];
  }
  o << "}},\n";
  return o.str();
}

}  // namespace

const std::array<std::string, 256>& bytes_to_unicode() {
  // GPT-2's encoder.py: printable bytes ('!'..'~', '¡'..'¬', '®'..'ÿ') map to themselves,
  // every other byte to U+0100 + n in byte order.
  static const std::array<std::string, 256> table = [] {
    std::array<std::string, 256> t;
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; ++b) {
      const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
      append_utf8(t[b], printable ? b : 256 + n++);
    }
    return t;
  }();
  return table;
}

std::string to_unicode_string(std::string_view bytes) {
  const auto& table = bytes_to_unicode();
  std::string out;
  out.reserve(bytes.size() * 2);
  for (unsigned char c : bytes) out += table[c];
  return out;
}

Vocab build_vocab(const std::vector<Merge>& merges,
                  const std::vector<std::string>& special_tokens) {
  Vocab vocab;
  vocab.reserve(kNumByteTokens + special_tokens.size() + merges.size());
  for (int i = 0; i < 256; ++i) vocab.emplace_back(1, static_cast<char>(i));
  for (const auto& s : special_tokens) vocab.push_back(s);
  for (const Merge& m : merges) {
    if (m.new_id != vocab.size() || m.a >= vocab.size() || m.b >= vocab.size()) {
      throw std::invalid_argument("build_vocab: merge ids out of order");
    }
    vocab.push_back(vocab[m.a] + vocab[m.b]);
  }
  return vocab;
}

std::string format_words(const WordCounts& counts) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  for (size_t i = 0; i < counts.words.size(); ++i) {
    for (unsigned char c : counts.words[i]) {
      out.push_back(kHex[c >> 4]);
      out.push_back(kHex[c & 0xF]);
    }
    out.push_back('\t');
    out += std::to_string(counts.freq[i]);
    out.push_back('\n');
  }
  return out;
}

std::string format_merges(const std::vector<Merge>& merges, const Vocab& vocab) {
  std::string out = "#version: 0.2\n";
  for (const Merge& m : merges) {
    out += to_unicode_string(vocab[m.a]);
    out += ' ';
    out += to_unicode_string(vocab[m.b]);
    out += '\n';
  }
  return out;
}

std::string json_string(std::string_view s) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          out += "\\u00";
          out += kHex[c >> 4];
          out += kHex[c & 0xF];
        } else {
          out += static_cast<char>(c);  // UTF-8 passes through raw (ensure_ascii=False)
        }
    }
  }
  out += '"';
  return out;
}

std::string format_vocab(const Vocab& vocab, size_t n_special) {
  ankerl::unordered_dense::set<std::string> seen;
  std::string out = "{";
  for (size_t id = 0; id < vocab.size(); ++id) {
    const bool special = id >= kNumByteTokens && id < kNumByteTokens + n_special;
    std::string key = special ? vocab[id] : to_unicode_string(vocab[id]);
    if (!seen.insert(key).second)
      throw std::runtime_error("vocab.json: duplicate key for id " + std::to_string(id));
    if (id > 0) out += ',';
    out += json_string(key);
    out += ':';
    out += std::to_string(id);
  }
  out += '}';
  return out;
}

std::string format_timings(const TrainResult& result, const std::string& input,
                           const TrainConfig& cfg) {
  const PhaseTimings& t = result.timings;
  const Hardware h = detect_hardware();
  std::ostringstream o;
  o << "{\n"
    << "  \"merges\": " << result.merges.size() << ",\n"
    << "  \"timings\": {\"chunk\": " << format_double(t.chunk)
    << ", \"pretok\": " << format_double(t.pretok) << ", \"build\": " << format_double(t.build)
    << ", \"pairs\": " << format_double(t.pairs) << ", \"merge\": " << format_double(t.merge)
    << ", \"total\": " << format_double(t.total) << "},\n";
  if (cfg.merge == MergeStrategy::IncrBatch) o << format_batch_stats(result.stats, cfg);
  if (cfg.merge == MergeStrategy::IncrPar) {
    const MergeStats& s = result.stats;
    o << "  \"owners\": {\"map\": \"" << name_of(cfg.owner_map) << '"';
    if (cfg.owner_stats) {
      const double ratio = s.owner_mean > 0 ? s.owner_busiest / s.owner_mean : 0;
      o << ", \"big_steps\": " << s.big_steps << ", \"busiest_mean\": " << format_double(ratio);
    }
    o << "},\n";
  }
  o << "  \"config\": {\"input\": " << json_string(input) << ", \"vocab_size\": " << cfg.vocab_size
    << ", \"threads\": " << cfg.threads << ", \"pretok\": \"" << name_of(cfg.pretok)
    << "\", \"pretok_reduce\": \"" << name_of(cfg.pretok_reduce) << "\", \"merge\": \""
    << name_of(cfg.merge) << "\", \"par_threshold_tokens\": " << cfg.par_threshold_tokens
    << ", \"work_unit_bytes\": " << cfg.work_unit_bytes << "},\n"
    << "  \"hardware\": {\"cpu_model\": " << json_string(h.cpu_model)
    << ", \"physical_cores\": " << h.physical_cores << ", \"logical_cores\": " << h.logical_cores
    << ", \"smt\": " << (h.logical_cores > h.physical_cores ? "true" : "false");
  if (h.performance_cores >= 0) {
    o << ", \"performance_cores\": " << h.performance_cores
      << ", \"efficiency_cores\": " << h.efficiency_cores;
  }
  o << ", \"os\": " << json_string(h.os) << "},\n"
    << "  \"git_sha\": " << json_string(git_sha()) << "\n"
    << "}\n";
  return o.str();
}

void write_file(const std::string& path, std::string_view contents) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!f) throw std::runtime_error("write failed: " + path);
}

}  // namespace parabpe
