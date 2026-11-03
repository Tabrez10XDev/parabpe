#pragma once

// Internal: pieces every merge engine shares, so they all implement SPEC.md §3–§4 identically.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "parabpe/types.hpp"

namespace parabpe::detail {

// Number of merges to learn: vocab_size − 256 − n_special (SPEC.md §6 id layout).
inline size_t num_merges(const TrainConfig& cfg) {
  const size_t base = kNumByteTokens + cfg.special_tokens.size();
  if (cfg.vocab_size < base) {
    throw std::invalid_argument("vocab_size " + std::to_string(cfg.vocab_size) + " < 256 + " +
                                std::to_string(cfg.special_tokens.size()) + " special tokens");
  }
  return cfg.vocab_size - base;
}

// Token id -> raw bytes, grown by one entry per merge. Used for the tie-break.
inline std::vector<std::string> initial_token_bytes(const TrainConfig& cfg) {
  std::vector<std::string> bytes;
  bytes.reserve(cfg.vocab_size);
  for (int i = 0; i < 256; ++i) bytes.emplace_back(1, static_cast<char>(i));
  for (const auto& s : cfg.special_tokens) bytes.push_back(s);
  return bytes;
}

// SPEC.md §4: is pair x (count cx) better than pair y (count cy)? Highest count wins; on a tie the
// lexicographically largest (bytes(a), bytes(b)) tuple wins. string_view compares bytes unsigned.
inline bool better_pair(uint64_t cx, PairKey x, uint64_t cy, PairKey y,
                        const std::vector<std::string>& token_bytes) {
  if (cx != cy) return cx > cy;
  const std::string_view xa = token_bytes[pair_first(x)], ya = token_bytes[pair_first(y)];
  if (const int c = xa.compare(ya); c != 0) return c > 0;
  return std::string_view(token_bytes[pair_second(x)]) >
         std::string_view(token_bytes[pair_second(y)]);
}

// SPEC.md §3: rewrites w[0, len) in place, merging (a, b) -> z greedily, left to right, without
// overlaps (a a a a -> z z; a a a -> z a). Returns the new length.
inline uint32_t apply_merge(TokenId* w, uint32_t len, TokenId a, TokenId b, TokenId z) {
  uint32_t out = 0;
  for (uint32_t i = 0; i < len;) {
    if (i + 1 < len && w[i] == a && w[i + 1] == b) {
      w[out++] = z;
      i += 2;
    } else {
      w[out++] = w[i++];
    }
  }
  return out;
}

}  // namespace parabpe::detail
