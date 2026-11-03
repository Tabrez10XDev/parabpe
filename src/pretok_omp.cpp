#include <omp.h>

#include <ankerl/unordered_dense.h>
#include <bit>
#include <cstdint>
#include <exception>
#include <optional>
#include <utility>
#include <vector>

#include "parabpe/regex.hpp"
#include "strategies.hpp"

namespace parabpe::detail {
namespace {

using WordMap = ankerl::unordered_dense::map<std::string_view, uint64_t>;

// Per-thread and per-shard state is written in hot loops: keep each on its own cache line.
struct alignas(64) LocalMaps {
  std::vector<WordMap> buckets;  // bucket b holds the words whose low hash bits are b
};
struct alignas(64) Shard {
  WordMap words;
};

// S4: dynamic schedule over work units; each thread has its own Matcher and local maps
// pre-bucketed by the low hash bits. After the barrier, bucket b of every thread is merged into
// shard b, so no two threads ever write the same map (no locks). PretokReduce::Critical instead
// merges one map per thread into a global map inside a critical section, for the report.
class OmpPreTokenizer final : public PreTokenizer {
 public:
  WordCounts run(const std::vector<WorkUnit>& units, const TrainConfig& cfg) override {
    const Regex re(cfg.pattern);
    const bool critical = cfg.pretok_reduce == PretokReduce::Critical;
    const int n_units = static_cast<int>(units.size());
    std::vector<LocalMaps> locals(cfg.threads);
    std::vector<Shard> shards;
    WordMap global;  // PretokReduce::Critical only
    std::exception_ptr error;

    // Exceptions must not leave the parallel region, and every thread must reach every
    // worksharing construct, so failures are recorded and rethrown afterwards.
    auto record = [&error]() {
#pragma omp critical(parabpe_pretok_error)
      if (!error) error = std::current_exception();
    };

#pragma omp parallel num_threads(cfg.threads)
    {
      const int n_threads = omp_get_num_threads();
      const unsigned n_buckets = critical ? 1u : std::bit_ceil(static_cast<unsigned>(n_threads));
      // Low bits pick the bucket: ankerl maps index their slots by the HIGH bits, so bucketing on
      // those would leave each local map using 1/n_buckets of its slots (measured: 50x slower at
      // 3 threads on 1 GB).
      const uint64_t mask = n_buckets - 1;
      LocalMaps& mine = locals[omp_get_thread_num()];
      mine.buckets.resize(n_buckets);

      std::optional<Matcher> matcher;
      try {
        matcher.emplace(re);
      } catch (...) {
        record();
      }

#pragma omp for schedule(dynamic, 1)
      for (int u = 0; u < n_units; ++u) {
        if (!matcher) continue;
        try {
          for (std::string_view doc : units[u].docs) {
            matcher->for_each_match(doc, [&](std::string_view word) {
              const uint64_t h = ankerl::unordered_dense::hash<std::string_view>{}(word);
              ++mine.buckets[h & mask][word];
            });
          }
        } catch (...) {
          record();
        }
      }  // implicit barrier: every local map is complete

      if (critical) {
#pragma omp critical(parabpe_pretok_merge)
        for (const auto& [word, n] : mine.buckets[0]) global[word] += n;
      } else {
#pragma omp single
        shards.resize(n_buckets);  // implicit barrier
#pragma omp for schedule(dynamic, 1)
        for (unsigned b = 0; b < n_buckets; ++b) {
          WordMap& shard = shards[b].words;
          shard = std::move(locals[0].buckets[b]);
          for (int t = 1; t < n_threads; ++t) {
            for (const auto& [word, n] : locals[t].buckets[b]) shard[word] += n;
          }
        }
      }
    }
    if (error) std::rethrow_exception(error);

    // Keys are views into the corpus, which outlives this call. Shards hold disjoint words.
    std::vector<std::pair<std::string_view, uint64_t>> entries;
    if (critical) {
      entries = std::move(global).extract();
    } else {
      size_t total = 0;
      for (const Shard& s : shards) total += s.words.size();
      entries.reserve(total);
      for (Shard& s : shards) {
        for (const auto& kv : s.words) entries.push_back(kv);
      }
    }
    return sorted_word_counts(std::move(entries));
  }
};

}  // namespace

std::unique_ptr<PreTokenizer> make_omp_pretokenizer() {
  return std::make_unique<OmpPreTokenizer>();
}

}  // namespace parabpe::detail
