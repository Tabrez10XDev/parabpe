#pragma once

// Helpers shared by the trace tests (test_trace, test_merge_threads, test_trace_words).

#include <cstddef>
#include <random>
#include <string>

#include "golden_case.hpp"  // test::read_file
#include "parabpe/trainer.hpp"
#include "trace.hpp"

namespace parabpe::test {

// n random letters, spaces and newlines, with a special token every 400 characters.
inline std::string random_text(unsigned seed, int n) {
  std::mt19937 rng(seed);
  const std::string alphabet = "abcdefgh  \n";
  std::string text;
  for (int i = 0; i < n; ++i) {
    text += alphabet[rng() % alphabet.size()];
    if (i % 400 == 399) text += "<|endoftext|>";
  }
  return text;
}

// Runs one training with `tracer` installed directly, so the test can read its events.
inline TrainResult traced(const std::string& text, const TrainConfig& cfg, detail::Tracer& tracer) {
  detail::g_tracer = &tracer;
  TrainResult r;
  try {
    r = train_text(text, cfg);
  } catch (...) {
    detail::g_tracer = nullptr;
    throw;
  }
  detail::g_tracer = nullptr;
  return r;
}

inline size_t occurrences(const std::string& s, const std::string& needle) {
  size_t n = 0;
  for (size_t p = s.find(needle); p != std::string::npos; p = s.find(needle, p + 1)) ++n;
  return n;
}

}  // namespace parabpe::test
