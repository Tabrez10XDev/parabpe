#include "parabpe/corpus.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace parabpe {

Corpus Corpus::open(const std::string& path) {
  Corpus c;
  c.fd_ = ::open(path.c_str(), O_RDONLY);
  if (c.fd_ < 0) throw std::runtime_error("cannot open " + path);

  struct stat st{};
  if (::fstat(c.fd_, &st) != 0) throw std::runtime_error("cannot stat " + path);
  c.size_ = static_cast<size_t>(st.st_size);
  if (c.size_ == 0) return c;

  void* p = ::mmap(nullptr, c.size_, PROT_READ, MAP_PRIVATE, c.fd_, 0);
  if (p == MAP_FAILED) throw std::runtime_error("cannot mmap " + path);
  ::madvise(p, c.size_, MADV_SEQUENTIAL);
  c.data_ = static_cast<const char*>(p);
  return c;
}

Corpus::~Corpus() {
  if (data_) ::munmap(const_cast<char*>(data_), size_);
  if (fd_ >= 0) ::close(fd_);
}

Corpus::Corpus(Corpus&& o) noexcept
    : data_(std::exchange(o.data_, nullptr)),
      size_(std::exchange(o.size_, 0)),
      fd_(std::exchange(o.fd_, -1)) {}

Corpus& Corpus::operator=(Corpus&& o) noexcept {
  if (this != &o) {
    this->~Corpus();
    data_ = std::exchange(o.data_, nullptr);
    size_ = std::exchange(o.size_, 0);
    fd_ = std::exchange(o.fd_, -1);
  }
  return *this;
}

void validate_utf8(std::string_view text) {
  const auto* s = reinterpret_cast<const unsigned char*>(text.data());
  const size_t n = text.size();
  size_t i = 0;
  auto fail = [&i]() {
    throw std::runtime_error("invalid UTF-8 at byte offset " + std::to_string(i));
  };
  while (i < n) {
    // ASCII fast path, 8 bytes at a time.
    if (i + 8 <= n) {
      uint64_t word;
      std::memcpy(&word, s + i, 8);
      if ((word & 0x8080808080808080ull) == 0) {
        i += 8;
        continue;
      }
    }
    const unsigned char c = s[i];
    if (c < 0x80) {
      ++i;
      continue;
    }
    // Lead byte -> sequence length and the allowed range of the second byte (RFC 3629 table).
    size_t len = 0;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) {
      len = 2;
    } else if (c >= 0xE0 && c <= 0xEF) {
      len = 3;
      if (c == 0xE0) lo = 0xA0;  // no overlongs
      if (c == 0xED) hi = 0x9F;  // no surrogates
    } else if (c >= 0xF0 && c <= 0xF4) {
      len = 4;
      if (c == 0xF0) lo = 0x90;  // no overlongs
      if (c == 0xF4) hi = 0x8F;  // nothing above U+10FFFF
    } else {
      fail();  // continuation byte, overlong C0/C1, or F5..FF
    }
    if (i + len > n || s[i + 1] < lo || s[i + 1] > hi) fail();
    for (size_t k = 2; k < len; ++k) {
      if ((s[i + k] & 0xC0) != 0x80) fail();
    }
    i += len;
  }
}

std::vector<WorkUnit> make_work_units(std::string_view text,
                                      const std::vector<std::string>& special_tokens,
                                      size_t target_bytes) {
  for (size_t k = 0; k < special_tokens.size(); ++k) {
    if (special_tokens[k].empty()) throw std::invalid_argument("special tokens must be non-empty");
    for (size_t j = 0; j < k; ++j) {
      if (special_tokens[j] == special_tokens[k]) {
        throw std::invalid_argument("duplicate special token: " + special_tokens[k]);
      }
    }
  }
  validate_utf8(text);

  std::vector<WorkUnit> units;
  WorkUnit current;
  size_t current_bytes = 0;
  auto close_current = [&]() {
    if (!current.docs.empty()) units.push_back(std::move(current));
    current = WorkUnit{};
    current_bytes = 0;
  };
  auto add_doc = [&](std::string_view doc) {
    if (doc.empty()) return;                          // consecutive / leading / trailing separators
    if (doc.size() >= target_bytes) close_current();  // a big document gets its own unit
    current.docs.push_back(doc);
    current_bytes += doc.size();
    if (current_bytes >= target_bytes) close_current();
  };

  // next[k] = next occurrence of special k at or after `pos` (npos = none left).
  constexpr size_t npos = std::string_view::npos;
  std::vector<size_t> next(special_tokens.size());
  for (size_t k = 0; k < special_tokens.size(); ++k) next[k] = text.find(special_tokens[k]);
  size_t pos = 0;
  while (true) {
    // Leftmost match; at the same position the longest special wins.
    size_t best = npos, best_len = 0;
    for (size_t k = 0; k < special_tokens.size(); ++k) {
      const size_t len = special_tokens[k].size();
      if (next[k] < best || (next[k] == best && next[k] != npos && len > best_len)) {
        best = next[k];
        best_len = len;
      }
    }
    if (best == npos) break;
    add_doc(text.substr(pos, best - pos));
    pos = best + best_len;
    for (size_t k = 0; k < special_tokens.size(); ++k) {
      if (next[k] != npos && next[k] < pos) next[k] = text.find(special_tokens[k], pos);
    }
  }
  add_doc(text.substr(pos));
  close_current();
  return units;
}

}  // namespace parabpe
