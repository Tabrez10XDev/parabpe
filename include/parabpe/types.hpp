#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace parabpe {

using TokenId = uint32_t;
using PairKey = uint64_t;  // (a << 32) | b

constexpr PairKey make_pair_key(TokenId a, TokenId b) {
  return (PairKey(a) << 32) | b;
}
constexpr TokenId pair_first(PairKey k) {
  return TokenId(k >> 32);
}
constexpr TokenId pair_second(PairKey k) {
  return TokenId(k & 0xffffffffu);
}

constexpr TokenId kNumByteTokens = 256;

// GPT-2 pre-tokenization pattern (SPEC.md §2).
inline constexpr const char* kGPT2Pattern =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

struct Merge {
  TokenId a, b, new_id;
  bool operator==(const Merge&) const = default;
};

enum class PretokStrategy { Serial, OMP };
// OMP pre-tokenizer: how thread-local word maps are combined (critical = the report's baseline).
enum class PretokReduce { Bucketed, Critical };
enum class MergeStrategy { NaiveSerial, IncrSerial, NaivePar, IncrPar, IncrBatch };

struct TrainConfig {
  size_t vocab_size = 32000;
  int threads = 1;
  std::string pattern = kGPT2Pattern;
  std::vector<std::string> special_tokens = {"<|endoftext|>"};
  PretokStrategy pretok = PretokStrategy::Serial;
  PretokReduce pretok_reduce = PretokReduce::Bucketed;
  MergeStrategy merge = MergeStrategy::IncrSerial;
  size_t par_threshold_tokens = 4096;  // IncrPar: below this, the master does the step alone
  size_t work_unit_bytes = 1 << 20;    // fixed, independent of thread count (SPEC.md §2)
  bool verify_batches = false;         // IncrBatch: check every member against a sequential oracle
  size_t owners_per_thread = 4;        // IncrBatch: hashed owners per thread (1 owner at 1 thread)
  size_t batch_scan_cap = 4096;        // IncrBatch: live tie-group entries scanned per round
  std::string trace_path;              // --trace: Chrome Trace JSON output, empty = off
};

// Output of phase 1: unique pre-tokens (raw bytes) and their frequencies.
struct WordCounts {
  std::vector<std::string> words;
  std::vector<uint64_t> freq;
};

// Struct-of-arrays word storage. Words only shrink, so merges rewrite in place.
struct WordTable {
  std::vector<TokenId> tokens;  // all words concatenated
  std::vector<uint32_t> offset, len, last_touched;
  std::vector<uint64_t> freq;

  size_t size() const { return offset.size(); }
};

struct PhaseTimings {
  double chunk = 0, pretok = 0, build = 0, pairs = 0, merge = 0, total = 0;
};

// Why a batched round stopped growing (docs/notes/batched_merging.md).
enum class BatchStop { Disjoint, Tie, Self, Budget, Capped };

// Batched-merging statistics; stays zero for the other engines.
struct MergeStats {
  uint64_t rounds = 0, max_batch = 0, parallel_rounds = 0, changed_pairs = 0;
  std::array<uint64_t, 9> batch_hist{};  // sizes 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65-128, 129+
  std::array<uint64_t, 5> stops{};       // indexed by BatchStop
  double select_s = 0, order_s = 0;      // serial selection and heap-update time in phase 3
};

// Token id -> raw bytes. Ids [0, 256) are single bytes, then special tokens, then merges.
using Vocab = std::vector<std::string>;

struct TrainResult {
  std::vector<Merge> merges;
  Vocab vocab;
  PhaseTimings timings;
  MergeStats stats;
};

}  // namespace parabpe
