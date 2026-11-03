#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace parabpe {

// Read-only, memory-mapped corpus file.
class Corpus {
 public:
  static Corpus open(const std::string& path);
  ~Corpus();
  Corpus(Corpus&&) noexcept;
  Corpus& operator=(Corpus&&) noexcept;
  Corpus(const Corpus&) = delete;
  Corpus& operator=(const Corpus&) = delete;

  std::string_view text() const { return {data_, size_}; }

 private:
  Corpus() = default;
  const char* data_ = nullptr;
  size_t size_ = 0;
  int fd_ = -1;
};

// A group of whole documents (special tokens already removed). See SPEC.md §2.
struct WorkUnit {
  std::vector<std::string_view> docs;
};

// Throws std::runtime_error (with the byte offset) unless text is valid UTF-8: no overlong forms,
// no surrogates, nothing above U+10FFFF, no truncated sequences. Same rules as Python's strict
// decode.
void validate_utf8(std::string_view text);

// Validates UTF-8, then splits text on special tokens (leftmost match, longest first), drops them
// and skips empty documents (SPEC.md §2). Documents are packed in order: a unit closes once its
// bytes reach target_bytes, and a document of at least target_bytes gets a unit of its own. The
// result does not depend on the thread count. Throws std::invalid_argument on empty or duplicate
// special tokens.
std::vector<WorkUnit> make_work_units(std::string_view text,
                                      const std::vector<std::string>& special_tokens,
                                      size_t target_bytes);

}  // namespace parabpe
