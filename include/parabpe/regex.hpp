#pragma once

#include <cstddef>
#include <string>
#include <string_view>

// PCRE2 handle types (8-bit code units), forward-declared so users don't need pcre2.h.
struct pcre2_real_code_8;
struct pcre2_real_match_data_8;
struct pcre2_real_match_context_8;
struct pcre2_real_jit_stack_8;

namespace parabpe {

// Rewrites \s -> \p{White_Space} and \S -> \P{White_Space}. With PCRE2_UCP, PCRE2's \s uses a
// legacy hard-coded list that also matches U+180E (Mongolian vowel separator), which stopped being
// whitespace in Unicode 6.3. \p{White_Space} comes from the Unicode 16 tables and matches Python's
// `regex` exactly (SPEC.md §2; checked per code point by test_unicode_parity.cpp).
// Not for patterns that use \Q...\E literal sections.
std::string unicode_whitespace_pattern(std::string_view pattern);

// A compiled, JIT-compiled pattern (PCRE2_UTF | PCRE2_UCP), with \s/\S rewritten by
// unicode_whitespace_pattern. Immutable, safe to share across threads.
class Regex {
 public:
  explicit Regex(const std::string& pattern);  // throws std::runtime_error on a bad pattern
  ~Regex();
  Regex(Regex&&) noexcept;
  Regex& operator=(Regex&&) noexcept;
  Regex(const Regex&) = delete;
  Regex& operator=(const Regex&) = delete;

  const pcre2_real_code_8* code() const { return code_; }

 private:
  pcre2_real_code_8* code_ = nullptr;
};

// Per-thread match state (match data + JIT stack). Not thread-safe; give each thread its own.
// Text must already be valid UTF-8: matching uses PCRE2_NO_UTF_CHECK (SPEC.md §1).
class Matcher {
 public:
  explicit Matcher(const Regex& re);
  ~Matcher();
  Matcher(const Matcher&) = delete;
  Matcher& operator=(const Matcher&) = delete;

  // End offset of the match that starts exactly at `pos`. Throws std::runtime_error on any PCRE2
  // error, on an empty match, or if the next match starts after `pos` (a gap would drop text).
  size_t match_at(std::string_view text, size_t pos);

  // Calls f(piece) for consecutive matches that tile `text` exactly (like Python's regex.findall
  // for patterns that cover every character, as the GPT-2 pattern does).
  template <class F>
  void for_each_match(std::string_view text, F&& f) {
    for (size_t pos = 0; pos < text.size();) {
      const size_t end = match_at(text, pos);
      f(text.substr(pos, end - pos));
      pos = end;
    }
  }

 private:
  const Regex* re_;
  pcre2_real_match_data_8* match_data_ = nullptr;
  pcre2_real_match_context_8* context_ = nullptr;
  pcre2_real_jit_stack_8* jit_stack_ = nullptr;
};

}  // namespace parabpe
