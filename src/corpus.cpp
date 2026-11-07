#include "parabpe/corpus.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "trace.hpp"

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

namespace {

// Offset of the first invalid sequence that starts in [begin, end), or npos. `begin` must be a
// sequence boundary; the last sequence may read past `end` (up to text.size()).
size_t first_invalid_utf8(std::string_view text, size_t begin, size_t end) {
  const auto* s = reinterpret_cast<const unsigned char*>(text.data());
  const size_t n = text.size();
  size_t i = begin;
  while (i < end) {
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
      return i;  // continuation byte, overlong C0/C1, or F5..FF
    }
    if (i + len > n || s[i + 1] < lo || s[i + 1] > hi) return i;
    for (size_t k = 2; k < len; ++k) {
      if ((s[i + k] & 0xC0) != 0x80) return i;
    }
    i += len;
  }
  return std::string_view::npos;
}

bool is_continuation(char c) {
  return (static_cast<unsigned char>(c) & 0xC0) == 0x80;
}

constexpr size_t kMinParallelBytes = 1 << 20;  // below this, threads cost more than they save

}  // namespace

void validate_utf8(std::string_view text, int threads) {
  constexpr size_t npos = std::string_view::npos;
  size_t bad = npos;
  if (threads <= 1 || text.size() < kMinParallelBytes) {
    bad = first_invalid_utf8(text, 0, text.size());
  } else {
    // Range r starts at the first non-continuation byte at or after r*n/T. In valid UTF-8 that is
    // a sequence boundary, so every range agrees with a serial scan up to the first error, and
    // the minimum over ranges is the serial answer for every T. Skipped continuation bytes belong
    // to the previous range, which reports them if they are invalid.
    const size_t n = text.size();
    std::vector<size_t> bounds(threads + 1, n);
    for (int r = 0; r < threads; ++r) {
      size_t b = n * r / threads;
      while (r > 0 && b < n && is_continuation(text[b])) ++b;
      bounds[r] = b;
    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) reduction(min : bad)
    for (int r = 0; r < threads; ++r) {
      bad = std::min(bad, first_invalid_utf8(text, bounds[r], std::max(bounds[r], bounds[r + 1])));
    }
  }
  if (bad != npos) throw std::runtime_error("invalid UTF-8 at byte offset " + std::to_string(bad));
}

namespace {

// Packs documents into work units in order (see make_work_units in corpus.hpp).
class UnitPacker {
 public:
  explicit UnitPacker(size_t target_bytes) : target_(target_bytes) {}
  void add(std::string_view doc) {
    if (doc.empty()) return;                     // consecutive / leading / trailing separators
    if (doc.size() >= target_) close_current();  // a big document gets its own unit
    current_.docs.push_back(doc);
    current_bytes_ += doc.size();
    if (current_bytes_ >= target_) close_current();
  }
  std::vector<WorkUnit> finish() {
    close_current();
    return std::move(units_);
  }

 private:
  void close_current() {
    if (!current_.docs.empty()) units_.push_back(std::move(current_));
    current_ = WorkUnit{};
    current_bytes_ = 0;
  }
  size_t target_;
  std::vector<WorkUnit> units_;
  WorkUnit current_;
  size_t current_bytes_ = 0;
};

void check_specials(const std::vector<std::string>& special_tokens) {
  for (size_t k = 0; k < special_tokens.size(); ++k) {
    if (special_tokens[k].empty()) throw std::invalid_argument("special tokens must be non-empty");
    for (size_t j = 0; j < k; ++j) {
      if (special_tokens[j] == special_tokens[k]) {
        throw std::invalid_argument("duplicate special token: " + special_tokens[k]);
      }
    }
  }
}

struct Occurrence {
  size_t pos, len;
};

// S10: every occurrence (overlapping ones included) of every special token, sorted by position
// and, at the same position, longest first. Thread r collects the occurrences that START in its
// byte range; searching up to range end + longest token - 1 finds the ones crossing the border.
std::vector<Occurrence> find_specials_parallel(std::string_view text,
                                               const std::vector<std::string>& specials,
                                               int threads) {
  const size_t n = text.size();
  size_t max_len = 0;
  for (const auto& s : specials) max_len = std::max(max_len, s.size());
  std::vector<std::vector<Occurrence>> found(threads);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
  for (int r = 0; r < threads; ++r) {
    const size_t begin = n * r / threads, end = n * (r + 1) / threads;
    const detail::TraceSpan span(detail::TraceKind::Chunk, end - begin);
    const std::string_view window = text.substr(begin, end - begin + max_len - 1);
    std::vector<Occurrence>& mine = found[r];
    for (const auto& s : specials) {
      for (size_t p = window.find(s); p != std::string_view::npos && begin + p < end;
           p = window.find(s, p + 1)) {
        mine.push_back({begin + p, s.size()});
      }
    }
    std::sort(mine.begin(), mine.end(), [](const Occurrence& x, const Occurrence& y) {
      return x.pos != y.pos ? x.pos < y.pos : x.len > y.len;
    });
  }
  std::vector<Occurrence> all;
  for (const auto& f : found) all.insert(all.end(), f.begin(), f.end());
  return all;
}

}  // namespace

std::vector<WorkUnit> make_work_units(std::string_view text,
                                      const std::vector<std::string>& special_tokens,
                                      size_t target_bytes, int threads) {
  check_specials(special_tokens);
  validate_utf8(text, threads);
  UnitPacker packer(target_bytes);
  constexpr size_t npos = std::string_view::npos;

  if (threads > 1 && text.size() >= kMinParallelBytes && !special_tokens.empty()) {
    // Same rule as the serial scan below, applied to the precomputed occurrence list: the first
    // occurrence at or after `pos` is the leftmost match, and at equal positions the longest
    // comes first.
    size_t pos = 0;
    for (const Occurrence& o : find_specials_parallel(text, special_tokens, threads)) {
      if (o.pos < pos) continue;  // overlaps the previous separator, or a shorter one at its start
      packer.add(text.substr(pos, o.pos - pos));
      pos = o.pos + o.len;
    }
    packer.add(text.substr(pos));
    return packer.finish();
  }

  // next[k] = next occurrence of special k at or after `pos` (npos = none left).
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
    packer.add(text.substr(pos, best - pos));
    pos = best + best_len;
    for (size_t k = 0; k < special_tokens.size(); ++k) {
      if (next[k] != npos && next[k] < pos) next[k] = text.find(special_tokens[k], pos);
    }
  }
  packer.add(text.substr(pos));
  return packer.finish();
}

}  // namespace parabpe
