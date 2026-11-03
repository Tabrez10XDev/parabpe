#include <omp.h>

#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bpe_common.hpp"
#include "strategies.hpp"

namespace parabpe::detail {
namespace {

using PairCounts = ankerl::unordered_dense::map<PairKey, int64_t>;
using PairIndex = ankerl::unordered_dense::map<PairKey, std::vector<uint32_t>>;  // pair -> word ids

struct HeapEntry {
  uint64_t count;
  PairKey pair;
};

bool contains_pair(const TokenId* tok, uint32_t len, TokenId a, TokenId b) {
  for (uint32_t i = 0; i + 1 < len; ++i) {
    if (tok[i] == a && tok[i + 1] == b) return true;
  }
  return false;
}

// Appends w unless it is already the last entry (a word visits its pairs in order, so repeats of
// the same pair within one word are consecutive). Older duplicates are filtered by last_touched.
void index_word(std::vector<uint32_t>& ids, uint32_t w) {
  if (ids.empty() || ids.back() != w) ids.push_back(w);
}

// S3 (serial): inverted index + whole-word diff updates + lazy max-heap.
// S6 (parallel): persistent omp region, owner-computes word partition with private
// indexes, padded per-thread delta buffers, threshold on tokens touched.
class IncrEngine final : public MergeEngine {
 public:
  explicit IncrEngine(bool parallel) : parallel_(parallel) {}

  std::vector<Merge> run(WordTable& table, const TrainConfig& cfg, PhaseTimings& t) override {
    // TODO(S6): parallel path.
    if (parallel_) throw std::logic_error("IncrPar: not implemented (S6)");

    const size_t n_merges = num_merges(cfg);
    std::vector<std::string> token_bytes = initial_token_bytes(cfg);
    std::vector<Merge> merges;
    merges.reserve(n_merges);

    // Heap order is SPEC.md §4. Token bytes never change once assigned, so stale entries keep a
    // stable position; they are dropped when popped (lazy deletion).
    auto heap_less = [&token_bytes](const HeapEntry& x, const HeapEntry& y) {
      return better_pair(y.count, y.pair, x.count, x.pair, token_bytes);
    };

    // Phase 2: counts, inverted index, heap.
    const double t0 = omp_get_wtime();
    PairCounts counts;
    PairIndex index;
    for (uint32_t w = 0; w < table.size(); ++w) {
      const TokenId* tok = table.tokens.data() + table.offset[w];
      for (uint32_t i = 0; i + 1 < table.len[w]; ++i) {
        const PairKey k = make_pair_key(tok[i], tok[i + 1]);
        counts[k] += static_cast<int64_t>(table.freq[w]);
        index_word(index[k], w);
      }
    }
    std::vector<HeapEntry> heap;
    auto rebuild_heap = [&]() {
      heap.clear();
      for (const auto& [k, c] : counts) heap.push_back({static_cast<uint64_t>(c), k});
      std::make_heap(heap.begin(), heap.end(), heap_less);
    };
    rebuild_heap();
    const double t1 = omp_get_wtime();
    t.pairs = t1 - t0;

    // Phase 3.
    PairCounts delta;
    for (uint32_t m = 0; m < n_merges; ++m) {
      // Pop until the top is live: its count is the current (positive) count of its pair.
      PairKey best = 0;
      bool found = false;
      while (!heap.empty()) {
        const HeapEntry top = heap.front();
        std::pop_heap(heap.begin(), heap.end(), heap_less);
        heap.pop_back();
        const auto it = counts.find(top.pair);
        if (it != counts.end() && static_cast<uint64_t>(it->second) == top.count) {
          best = top.pair;
          found = true;
          break;
        }
      }
      if (!found) break;  // nothing left to merge

      const TokenId a = pair_first(best), b = pair_second(best);
      const TokenId z = static_cast<TokenId>(token_bytes.size());
      // Before any push: the heap comparator reads the bytes of pairs that contain z.
      token_bytes.push_back(token_bytes[a] + token_bytes[b]);
      // Take the word list out first: inserting new pairs below may move the map's storage.
      const auto idx = index.find(best);
      const std::vector<uint32_t> words = std::move(idx->second);
      index.erase(idx);

      // Whole-word diff: subtract every pair of the old word, add every pair of the new one.
      // z is a fresh id, so the pairs new to a word are exactly the ones that contain z.
      delta.clear();
      for (const uint32_t w : words) {
        if (table.last_touched[w] == m) continue;
        table.last_touched[w] = m;
        TokenId* tok = table.tokens.data() + table.offset[w];
        const uint32_t len = table.len[w];
        if (!contains_pair(tok, len, a, b)) continue;
        const auto f = static_cast<int64_t>(table.freq[w]);
        for (uint32_t i = 0; i + 1 < len; ++i) delta[make_pair_key(tok[i], tok[i + 1])] -= f;
        const uint32_t new_len = apply_merge(tok, len, a, b, z);
        table.len[w] = new_len;
        for (uint32_t i = 0; i + 1 < new_len; ++i) {
          const PairKey k = make_pair_key(tok[i], tok[i + 1]);
          delta[k] += f;
          if (tok[i] == z || tok[i + 1] == z) index_word(index[k], w);
        }
      }

      for (const auto& [k, d] : delta) {
        if (d == 0) continue;
        const int64_t c = (counts[k] += d);
        if (c > 0) {
          heap.push_back({static_cast<uint64_t>(c), k});
          std::push_heap(heap.begin(), heap.end(), heap_less);
        } else if (c == 0) {
          counts.erase(k);
          index.erase(k);  // no word holds the pair any more
        } else {
          throw std::logic_error("IncrSerial: negative pair count");
        }
      }
      if (heap.size() > 4 * counts.size()) rebuild_heap();

      merges.push_back({a, b, z});
    }
    t.merge = omp_get_wtime() - t1;
    return merges;
  }

 private:
  bool parallel_;
};

}  // namespace

std::unique_ptr<MergeEngine> make_incr_engine(bool parallel) {
  return std::make_unique<IncrEngine>(parallel);
}

}  // namespace parabpe::detail
