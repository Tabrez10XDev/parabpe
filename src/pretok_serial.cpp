#include <algorithm>
#include <ankerl/unordered_dense.h>

#include "parabpe/regex.hpp"
#include "strategies.hpp"

namespace parabpe::detail {

WordCounts sorted_word_counts(std::vector<std::pair<std::string_view, uint64_t>> entries) {
  // string_view compares through char_traits<char>, i.e. as unsigned bytes, like Python's bytes.
  std::sort(entries.begin(), entries.end(),
            [](const auto& x, const auto& y) { return x.first < y.first; });
  WordCounts out;
  out.words.reserve(entries.size());
  out.freq.reserve(entries.size());
  for (const auto& [word, freq] : entries) {
    out.words.emplace_back(word);
    out.freq.push_back(freq);
  }
  return out;
}

namespace {

// S1: PCRE2 (JIT) split of each doc, counted into one map<bytes, count>.
class SerialPreTokenizer final : public PreTokenizer {
 public:
  WordCounts run(const std::vector<WorkUnit>& units, const TrainConfig& cfg) override {
    const Regex re(cfg.pattern);
    Matcher matcher(re);
    // Keys are views into the corpus (mmap), which outlives this call.
    ankerl::unordered_dense::map<std::string_view, uint64_t> counts;
    for (const WorkUnit& unit : units) {
      for (std::string_view doc : unit.docs) {
        matcher.for_each_match(doc, [&counts](std::string_view word) { ++counts[word]; });
      }
    }
    return sorted_word_counts(std::move(counts).extract());
  }
};

}  // namespace

std::unique_ptr<PreTokenizer> make_serial_pretokenizer() {
  return std::make_unique<SerialPreTokenizer>();
}

}  // namespace parabpe::detail
