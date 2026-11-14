#include "trace_words.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "incr_common.hpp"

namespace parabpe::detail {

std::vector<TraceWord> trace_word_sample(const WordCounts& counts, MergeStrategy merge,
                                         const TrainConfig& merge_cfg, size_t n) {
  std::vector<uint32_t> idx;
  for (uint32_t w = 0; w < counts.words.size(); ++w) {
    if (counts.words[w].size() >= kTraceWordMinBytes) idx.push_back(w);
  }
  const auto better = [&](uint32_t a, uint32_t b) {
    return counts.freq[a] != counts.freq[b] ? counts.freq[a] > counts.freq[b] : a < b;
  };
  const size_t keep = std::min(n, idx.size());
  std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(keep), idx.end(),
                    better);
  idx.resize(keep);

  const size_t owners = batch_owner_count(merge_cfg);
  std::vector<TraceWord> out;
  out.reserve(keep);
  for (const uint32_t w : idx) {
    const int64_t owner =
        merge == MergeStrategy::IncrBatch ? static_cast<int64_t>(hashed_owner(w, owners)) : -1;
    out.push_back({w, counts.freq[w], owner});
  }
  return out;
}

void write_trace_words(const std::string& path, const WordCounts& counts, MergeStrategy merge,
                       const TrainConfig& merge_cfg) {
  const auto sample = trace_word_sample(counts, merge, merge_cfg);
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write trace words file " + path);
  const bool batch = merge == MergeStrategy::IncrBatch;
  out << "{\"owners\": " << (batch ? batch_owner_count(merge_cfg) : 0)
      << ", \"merge_threads\": " << merge_cfg.threads << ", \"min_bytes\": " << kTraceWordMinBytes
      << ",\n\"words\": [";
  static constexpr char kHex[] = "0123456789abcdef";
  for (size_t i = 0; i < sample.size(); ++i) {
    const TraceWord& t = sample[i];
    out << (i ? ",\n" : "\n") << "{\"w\": " << t.w << ", \"count\": " << t.count
        << ", \"owner\": " << t.owner << ", \"hex\": \"";
    for (const unsigned char c : counts.words[t.w]) out << kHex[c >> 4] << kHex[c & 15];
    out << "\"}";
  }
  out << "\n]}\n";
  if (!out) throw std::runtime_error("error writing trace words file " + path);
}

}  // namespace parabpe::detail
