#pragma once

// Internal: the pure parts of batched merging, separate so tests can reach them directly. The rule
// and its proof are in docs/notes/batched_merging.md; all indices are 0-based, as they are there.

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "parabpe/types.hpp"

namespace parabpe::detail {

struct Cand {
  uint64_t count;
  PairKey pair;
};

// Each rule extends the one before it; only Closed ships, the others exist to be verified.
enum class BatchRule { Strict, Relaxed, Closed };

struct BatchChoice {
  size_t k;
  BatchStop stop;
};

// Members kept for a complete tie group T, where j is the first member whose tokens appear in T
// (K if none) and floor_k = max(g, 1). Growing T only lowers j, so k only shrinks, never below
// floor_k: a scan can stop as soon as this reaches floor_k, the rest of T cannot change the answer.
inline size_t keep_count(BatchRule rule, size_t K, size_t j, size_t floor_k, bool tail_empty) {
  switch (rule) {
    case BatchRule::Strict: return tail_empty ? K : floor_k;
    case BatchRule::Relaxed: return std::max(std::min(K, j), floor_k);
    case BatchRule::Closed: break;
  }
  return std::max(std::min(K, j + 1), floor_k);
}

// prefix: the best-first, pairwise token-disjoint members p[0..K), not empty. tail: every scanned
// live pair after p[K-1] with count c* = count(p[K-1]), including the entry that stopped the
// prefix. complete: the scan reached the end of the tie group, or stopped once keep_count fell to
// floor_k. tok_mark holds at least one slot per
// token and stamp is unused in it; tail tokens are stamped. prefix_stop is reported when k == K.
inline BatchChoice choose_batch(std::span<const Cand> prefix, std::span<const Cand> tail,
                                bool complete, std::vector<uint32_t>& tok_mark, uint32_t stamp,
                                BatchRule rule = BatchRule::Closed,
                                BatchStop prefix_stop = BatchStop::Disjoint) {
  const size_t K = prefix.size();
  const uint64_t c_star = prefix.back().count;
  size_t g = 0;  // members that beat every outside pair on count, so T cannot affect them
  for (const Cand& c : prefix) g += c.count > c_star;
  const size_t floor_k = std::max<size_t>(g, 1);

  size_t k = floor_k;
  if (complete) {
    for (const Cand& c : tail) tok_mark[pair_first(c.pair)] = tok_mark[pair_second(c.pair)] = stamp;
    size_t j = K;  // first member whose tokens appear in T
    for (size_t i = 0; i < K; ++i) {
      if (tok_mark[pair_first(prefix[i].pair)] == stamp ||
          tok_mark[pair_second(prefix[i].pair)] == stamp) {
        j = i;
        break;
      }
    }
    k = keep_count(rule, K, j, floor_k, tail.empty());
  }
  if (k == K) return {k, prefix_stop};
  return {k, complete ? BatchStop::Tie : BatchStop::Capped};
}

// One round's members, indexed for the single pass over a word. Member slot i merges into
// z_first + i. Tokens of different members never coincide, so a token starts at most one member.
struct Batch {
  std::vector<int32_t> first_slot;  // token -> slot of the member it starts, or -1
  std::vector<TokenId> second;      // slot -> second token of the member
  std::vector<PairKey> pairs;       // slot -> member
  TokenId z_first = 0;
  uint32_t stamp = 0;  // last_touched value for this round

  // Replaces the previous round's members. n_tokens counts the new tokens too.
  void assign(std::span<const Cand> members, TokenId first_new, size_t n_tokens, uint32_t round) {
    for (const PairKey p : pairs) first_slot[pair_first(p)] = -1;
    first_slot.resize(n_tokens, -1);
    pairs.clear();
    second.clear();
    for (const Cand& c : members) {
      first_slot[pair_first(c.pair)] = static_cast<int32_t>(pairs.size());
      second.push_back(pair_second(c.pair));
      pairs.push_back(c.pair);
    }
    z_first = first_new;
    stamp = round;
  }

  bool matches(TokenId a, TokenId b) const {
    const int32_t slot = first_slot[a];
    return slot >= 0 && second[slot] == b;
  }
};

inline bool contains_member(const TokenId* w, uint32_t len, const Batch& batch) {
  for (uint32_t i = 0; i + 1 < len; ++i) {
    if (batch.matches(w[i], w[i + 1])) return true;
  }
  return false;
}

// Rewrites w[0, len) in place with every member at once: the same result as apply_merge with each
// member in turn, because disjoint merges commute within a word and only a self-pair can overlap
// itself, which this left-to-right pass resolves like the greedy rule. Returns the new length.
inline uint32_t apply_batch(TokenId* w, uint32_t len, const Batch& batch) {
  uint32_t out = 0;
  for (uint32_t i = 0; i < len;) {
    if (i + 1 < len && batch.matches(w[i], w[i + 1])) {
      w[out++] = batch.z_first + static_cast<TokenId>(batch.first_slot[w[i]]);
      i += 2;
    } else {
      w[out++] = w[i++];
    }
  }
  return out;
}

}  // namespace parabpe::detail
