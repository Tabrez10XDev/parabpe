#include "parabpe/regex.hpp"

#include <pcre2.h>

#include <stdexcept>
#include <utility>

namespace parabpe {
namespace {

// JIT stack for long runs (e.g. a 1 MB whitespace block). Grows on demand up to the max.
constexpr size_t kJitStackStart = 64 * 1024;
constexpr size_t kJitStackMax = 16 * 1024 * 1024;

std::string error_message(int code) {
  PCRE2_UCHAR buf[256];
  const int n = pcre2_get_error_message(code, buf, sizeof(buf));
  return n < 0 ? "PCRE2 error " + std::to_string(code) : std::string(reinterpret_cast<char*>(buf));
}

}  // namespace

std::string unicode_whitespace_pattern(std::string_view pattern) {
  std::string out;
  out.reserve(pattern.size() + 32);
  for (size_t i = 0; i < pattern.size(); ++i) {
    if (pattern[i] == '\\' && i + 1 < pattern.size()) {
      const char next = pattern[++i];
      if (next == 's') {
        out += "\\p{White_Space}";
      } else if (next == 'S') {
        out += "\\P{White_Space}";
      } else {
        out += '\\';
        out += next;  // any other escape (including "\\") is copied unchanged
      }
    } else {
      out += pattern[i];
    }
  }
  return out;
}

Regex::Regex(const std::string& pattern) {
  int err = 0;
  PCRE2_SIZE erroff = 0;
  const std::string compiled = unicode_whitespace_pattern(pattern);
  code_ = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(compiled.data()), compiled.size(),
                        PCRE2_UTF | PCRE2_UCP, &err, &erroff, nullptr);
  if (!code_) {
    throw std::runtime_error("regex compile error at offset " + std::to_string(erroff) + ": " +
                             error_message(err));
  }
  // Falls back to the interpreter if JIT is unavailable on this platform; results are identical.
  pcre2_jit_compile(code_, PCRE2_JIT_COMPLETE);
}

Regex::~Regex() {
  pcre2_code_free(code_);
}

Regex::Regex(Regex&& o) noexcept : code_(std::exchange(o.code_, nullptr)) {}

Regex& Regex::operator=(Regex&& o) noexcept {
  if (this != &o) {
    pcre2_code_free(code_);
    code_ = std::exchange(o.code_, nullptr);
  }
  return *this;
}

Matcher::Matcher(const Regex& re) : re_(&re) {
  match_data_ = pcre2_match_data_create_from_pattern(re.code(), nullptr);
  context_ = pcre2_match_context_create(nullptr);
  jit_stack_ = pcre2_jit_stack_create(kJitStackStart, kJitStackMax, nullptr);
  if (!match_data_ || !context_ || !jit_stack_) {
    pcre2_jit_stack_free(jit_stack_);
    pcre2_match_context_free(context_);
    pcre2_match_data_free(match_data_);
    throw std::bad_alloc();
  }
  pcre2_jit_stack_assign(context_, nullptr, jit_stack_);
}

Matcher::~Matcher() {
  pcre2_jit_stack_free(jit_stack_);
  pcre2_match_context_free(context_);
  pcre2_match_data_free(match_data_);
}

size_t Matcher::match_at(std::string_view text, size_t pos) {
  const int rc = pcre2_match(re_->code(), reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(),
                             pos, PCRE2_NO_UTF_CHECK, match_data_, context_);
  if (rc == PCRE2_ERROR_NOMATCH) {
    throw std::runtime_error("regex: no match at offset " + std::to_string(pos));
  }
  if (rc < 0) {
    throw std::runtime_error("regex: match failed at offset " + std::to_string(pos) + ": " +
                             error_message(rc));
  }
  const PCRE2_SIZE* ov = pcre2_get_ovector_pointer(match_data_);
  if (ov[0] != pos) {
    throw std::runtime_error("regex: gap at offset " + std::to_string(pos) + " (next match at " +
                             std::to_string(ov[0]) + ")");
  }
  if (ov[1] == ov[0])
    throw std::runtime_error("regex: empty match at offset " + std::to_string(pos));
  return ov[1];
}

}  // namespace parabpe
