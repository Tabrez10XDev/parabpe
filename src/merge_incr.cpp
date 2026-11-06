#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "incr_common.hpp"
#include "strategies.hpp"

namespace parabpe::detail {
namespace {

// Adds one step's load per owner to `s`, counting what merge_owner will rewrite: each listed word
// once (`seen`, stamped with stamp), and only if it still holds the pair. Runs before the step.
void record_owner_load(const WordTable& table, const std::vector<Owner>& owners,
                       const MergeStep& step, std::vector<uint32_t>& seen, MergeStats& s) {
  const uint32_t stamp = step.m + 1;
  uint64_t busiest = 0, total = 0;
  for (const Owner& owner : owners) {
    const auto idx = owner.index.find(step.pair);
    if (idx == owner.index.end()) continue;
    uint64_t tokens = 0;
    for (const uint32_t w : idx->second) {
      if (seen[w] == stamp) continue;
      seen[w] = stamp;
      if (contains_pair(table.tokens.data() + table.offset[w], table.len[w], step.a, step.b)) {
        tokens += table.len[w];
      }
    }
    busiest = std::max(busiest, tokens);
    total += tokens;
  }
  if (total < kBigStepTokens) return;
  ++s.big_steps;
  s.owner_busiest += static_cast<double>(busiest);
  s.owner_mean += static_cast<double>(total) / static_cast<double>(owners.size());
}

// S3 (serial), S5 (parallel phase 2) and S6 (parallel phase 3): inverted index + whole-word diff
// updates + lazy max-heap. In IncrPar the master runs the merge loop; a step that touches at least
// cfg.par_threshold_tokens tokens opens a parallel region in which each owner merges its own words
// into padded per-thread delta buffers, shard s then sums and applies its deltas, and owners drop
// pairs that reached 0. Smaller steps run on the master alone, like IncrSerial. (A persistent
// region would cost a barrier on every one of the ~30k merges, most of which touch a handful of
// words.)
class IncrEngine final : public MergeEngine {
 public:
  explicit IncrEngine(bool parallel) : parallel_(parallel) {}

  std::vector<Merge> run(WordTable& table, const TrainConfig& cfg, PhaseTimings& t) override {
    const size_t n_merges = num_merges(cfg);
    std::vector<std::string> token_bytes = initial_token_bytes(cfg);
    std::vector<Merge> merges;
    merges.reserve(n_merges);

    // Heap order is SPEC.md §4. Token bytes never change once assigned, so stale entries keep a
    // stable position; they are dropped when popped (lazy deletion).
    auto heap_less = [&token_bytes](const HeapEntry& x, const HeapEntry& y) {
      return better_pair(y.count, y.pair, x.count, x.pair, token_bytes);
    };

    // Phase 2: counts, per-owner inverted indexes, heap.
    const double t0 = omp_get_wtime();
    std::vector<Owner> owners = parallel_ && cfg.owner_map == OwnerMap::Hashed
                                    ? partition_words_hashed(table, cfg.threads)
                                    : partition_words(table, parallel_ ? cfg.threads : 1);
    ShardedCounts counts;
    if (parallel_) {
      count_parallel(table, owners, counts, cfg.threads);
    } else {
      counts.reset(1);
      count_serial(table, owners[0], counts);
    }
    std::vector<HeapEntry> heap;
    auto rebuild_heap = [&]() {
      heap.clear();
      heap.reserve(counts.size());
      for (size_t s = 0; s < counts.n_shards(); ++s) {
        for (const auto& [k, c] : counts.shard(s)) heap.push_back({static_cast<uint64_t>(c), k});
      }
      std::make_heap(heap.begin(), heap.end(), heap_less);
    };
    rebuild_heap();
    const double t1 = omp_get_wtime();
    t.pairs = t1 - t0;

    auto push = [&](uint64_t c, PairKey k) {
      heap.push_back({c, k});
      std::push_heap(heap.begin(), heap.end(), heap_less);
    };

    // Phase 3.
    PairCounts delta;  // serial steps
    std::vector<ThreadDelta> thread_delta(parallel_ ? cfg.threads : 0);
    std::vector<ShardDelta> shard_delta(counts.n_shards());
    const bool log_owners = parallel_ && cfg.owner_stats;
    std::vector<uint32_t> seen(log_owners ? table.size() : 0, 0);
    for (uint32_t m = 0; m < n_merges; ++m) {
      // Pop until the top is live: its count is the current (positive) count of its pair.
      PairKey best = 0;
      bool found = false;
      while (!heap.empty()) {
        const HeapEntry top = heap.front();
        std::pop_heap(heap.begin(), heap.end(), heap_less);
        heap.pop_back();
        const int64_t* c = counts.find(top.pair);
        if (c && static_cast<uint64_t>(*c) == top.count) {
          best = top.pair;
          found = true;
          break;
        }
      }
      if (!found) break;  // nothing left to merge

      const MergeStep step{m, best, pair_first(best), pair_second(best),
                           static_cast<TokenId>(token_bytes.size())};
      // Before any push: the heap comparator reads the bytes of pairs that contain z.
      token_bytes.push_back(token_bytes[step.a] + token_bytes[step.b]);
      if (log_owners) record_owner_load(table, owners, step, seen, stats_);

      const bool par_step =
          parallel_ && cfg.threads > 1 &&
          tokens_touched(table, owners, best, cfg.par_threshold_tokens) >= cfg.par_threshold_tokens;
      if (par_step) {
        parallel_step(
            table, owners, counts,
            [&step](WordTable& tb, Owner& o, auto&& add) { merge_owner(tb, o, step, add); },
            thread_delta, shard_delta, cfg.threads);
        for (const ShardDelta& sd : shard_delta) {
          if (sd.negative) throw std::logic_error("IncrEngine: negative pair count");
          for (const HeapEntry& e : sd.push) push(e.count, e.pair);
        }
      } else {
        delta.clear();
        for (Owner& owner : owners) {
          merge_owner(table, owner, step, [&delta](PairKey k, int64_t d) { delta[k] += d; });
        }
        for (const auto& [k, d] : delta) {
          if (d == 0) continue;
          const int64_t c = counts.add(k, d);
          if (c > 0) {
            push(static_cast<uint64_t>(c), k);
          } else if (c == 0) {
            counts.erase(k);
            for (Owner& owner : owners) owner.index.erase(k);  // no word holds the pair any more
          } else {
            throw std::logic_error("IncrEngine: negative pair count");
          }
        }
      }
      if (heap.size() > 4 * counts.size()) rebuild_heap();

      merges.push_back({step.a, step.b, step.z});
    }
    t.merge = omp_get_wtime() - t1;
    return merges;
  }

  MergeStats stats() const override { return stats_; }

 private:
  bool parallel_;
  MergeStats stats_;
};

}  // namespace

std::unique_ptr<MergeEngine> make_incr_engine(bool parallel) {
  return std::make_unique<IncrEngine>(parallel);
}

}  // namespace parabpe::detail
