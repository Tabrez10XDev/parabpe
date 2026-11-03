#include <limits>
#include <stdexcept>

#include "parabpe/merge_engine.hpp"

namespace parabpe {

WordTable build_word_table(const WordCounts& counts) {
  WordTable t;
  const size_t n = counts.words.size();
  size_t total = 0;
  for (const auto& w : counts.words) total += w.size();
  if (n > std::numeric_limits<uint32_t>::max() || total > std::numeric_limits<uint32_t>::max()) {
    throw std::length_error("build_word_table: more than 2^32 words or tokens");
  }
  t.tokens.reserve(total);
  t.offset.reserve(n);
  t.len.reserve(n);
  t.freq = counts.freq;
  for (const auto& w : counts.words) {
    t.offset.push_back(static_cast<uint32_t>(t.tokens.size()));
    t.len.push_back(static_cast<uint32_t>(w.size()));
    for (unsigned char c : w) t.tokens.push_back(c);  // byte b -> token id b
  }
  t.last_touched.assign(n, std::numeric_limits<uint32_t>::max());
  return t;
}

}  // namespace parabpe
