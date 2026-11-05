#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "batch_select.hpp"
#include "incr_common.hpp"
#include "strategies.hpp"

namespace parabpe::detail {
namespace {

// Rewrites the owner's words that hold a member of the batch, with one whole-word diff per word per
// round, and reports the pair-count changes through add(pair, delta). The new tokens are fresh, so
// the pairs new to a word are exactly the ones holding a token >= z_first.
template <class Add>
void merge_owner_batch(WordTable& table, Owner& owner, const Batch& batch, Add&& add) {
  for (const PairKey member : batch.pairs) {
    const auto idx = owner.index.find(member);
    if (idx == owner.index.end()) continue;
    // Take the word list out first: inserting new pairs below may move the map's storage.
    const std::vector<uint32_t> words = std::move(idx->second);
    owner.index.erase(idx);
    for (const uint32_t w : words) {
      if (table.last_touched[w] == batch.stamp) continue;
      table.last_touched[w] = batch.stamp;
      TokenId* tok = table.tokens.data() + table.offset[w];
      const uint32_t len = table.len[w];
      if (!contains_member(tok, len, batch)) continue;
      const auto f = static_cast<int64_t>(table.freq[w]);
      for (uint32_t i = 0; i + 1 < len; ++i) add(make_pair_key(tok[i], tok[i + 1]), -f);
      const uint32_t new_len = apply_batch(tok, len, batch);
      table.len[w] = new_len;
      for (uint32_t i = 0; i + 1 < new_len; ++i) {
        const PairKey k = make_pair_key(tok[i], tok[i + 1]);
        add(k, f);
        if (tok[i] >= batch.z_first || tok[i + 1] >= batch.z_first) {
          index_word(owner.index[k], w);
        }
      }
    }
  }
}

// Does the round rewrite at least `threshold` tokens? Stops counting as soon as it knows.
bool reaches_threshold(const WordTable& table, const std::vector<Owner>& owners,
                       std::span<const Cand> members, size_t threshold) {
  size_t tokens = 0;
  for (const Cand& member : members) {
    if (tokens >= threshold) break;
    tokens += tokens_touched(table, owners, member.pair, threshold - tokens);
  }
  return tokens >= threshold;
}

// Drops pairs that reached 0 from every owner. Deferred, because each pair costs one lookup per
// owner. A pair at 0 stays at 0 (batched_merging.md), so its stale index entries are never read.
void sweep_zero_pairs(std::vector<Owner>& owners, std::vector<PairKey>& zero, int threads) {
  const int n_owners = static_cast<int>(owners.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
  for (int o = 0; o < n_owners; ++o) {
    for (const PairKey k : zero) owners[o].index.erase(k);
  }
  zero.clear();
}

// Lazy max-heap in SPEC.md §4 order over the pairs in `counts`.
class PairHeap {
 public:
  PairHeap(ShardedCounts& counts, const std::vector<std::string>& token_bytes)
      : counts_(counts), less_{&token_bytes} {}

  void rebuild() {
    heap_.clear();
    heap_.reserve(counts_.size());
    for (size_t s = 0; s < counts_.n_shards(); ++s) {
      for (const auto& [k, c] : counts_.shard(s)) {
        heap_.push_back({static_cast<uint64_t>(c), k});
      }
    }
    std::make_heap(heap_.begin(), heap_.end(), less_);
  }

  void push(uint64_t count, PairKey pair) {
    heap_.push_back({count, pair});
    std::push_heap(heap_.begin(), heap_.end(), less_);
  }

  // Pops the best live entry. A pair can have several identical live entries (its count went away
  // and came back); they are adjacent, so they are dropped here. A stray duplicate could only make
  // a batch smaller: a member's own entry would conflict with it, and goes stale once it merges.
  bool pop_live(Cand& out) {
    while (!heap_.empty()) {
      const HeapEntry top = pop_top();
      const int64_t* c = counts_.find(top.pair);
      if (!c || static_cast<uint64_t>(*c) != top.count) continue;
      while (!heap_.empty() && heap_.front().pair == top.pair && heap_.front().count == top.count) {
        pop_top();
      }
      out = {top.count, top.pair};
      return true;
    }
    return false;
  }

  size_t size() const { return heap_.size(); }

 private:
  struct Less {
    const std::vector<std::string>* token_bytes;
    bool operator()(const HeapEntry& x, const HeapEntry& y) const {
      return better_pair(y.count, y.pair, x.count, x.pair, *token_bytes);
    }
  };

  HeapEntry pop_top() {
    const HeapEntry top = heap_.front();
    std::pop_heap(heap_.begin(), heap_.end(), less_);
    heap_.pop_back();
    return top;
  }

  ShardedCounts& counts_;
  Less less_;
  std::vector<HeapEntry> heap_;
};

// --verify-batches: an IncrSerial replay on its own copy of the table. next() is the merge that
// sequential BPE makes at this point.
class SequentialOracle {
 public:
  SequentialOracle(const WordTable& table, const TrainConfig& cfg)
      : table_(table), token_bytes_(initial_token_bytes(cfg)), heap_(counts_, token_bytes_) {
    owners_ = partition_words(table_, 1);
    counts_.reset(1);
    count_serial(table_, owners_[0], counts_);
    heap_.rebuild();
  }

  bool next(Cand& best) {
    if (!heap_.pop_live(best)) return false;
    const MergeStep step{m_++, best.pair, pair_first(best.pair), pair_second(best.pair),
                         static_cast<TokenId>(token_bytes_.size())};
    token_bytes_.push_back(token_bytes_[step.a] + token_bytes_[step.b]);
    delta_.clear();
    merge_owner(table_, owners_[0], step, [this](PairKey k, int64_t d) { delta_[k] += d; });
    for (const auto& [k, d] : delta_) {
      if (d == 0) continue;
      const int64_t c = counts_.add(k, d);
      if (c > 0) {
        heap_.push(static_cast<uint64_t>(c), k);
      } else {
        counts_.erase(k);
        owners_[0].index.erase(k);
      }
    }
    if (heap_.size() > 4 * counts_.size()) heap_.rebuild();
    return true;
  }

 private:
  WordTable table_;
  std::vector<std::string> token_bytes_;
  ShardedCounts counts_;
  PairHeap heap_;
  std::vector<Owner> owners_;
  PairCounts delta_;
  uint32_t m_ = 0;
};

// Batched merge rounds (docs/notes/batched_merging.md): each round takes a prefix of the best-first
// order that is proven identical to the next k sequential merges, and applies it in one step. The
// master selects and updates the order. A round that rewrites at least cfg.par_threshold_tokens
// tokens runs on hashed owners in a parallel region (B4); smaller rounds run on the master alone.
class BatchEngine final : public MergeEngine {
 public:
  explicit BatchEngine(BatchRule rule) : rule_(rule) {}

  std::vector<Merge> run(WordTable& table, const TrainConfig& cfg, PhaseTimings& t) override {
    stats_ = {};
    const size_t n_merges = num_merges(cfg);
    token_bytes_ = initial_token_bytes(cfg);
    std::vector<Merge> merges;
    merges.reserve(n_merges);

    // Phase 2.
    const double t0 = omp_get_wtime();
    const bool par = cfg.threads > 1;
    std::vector<Owner> owners;
    ShardedCounts counts;
    if (par) {
      owners = partition_words_hashed(table, cfg.owners_per_thread * cfg.threads);
      count_parallel(table, owners, counts, cfg.threads);
    } else {
      owners = partition_words(table, 1);
      counts.reset(1);
      count_serial(table, owners[0], counts);
    }
    PairHeap heap(counts, token_bytes_);
    heap.rebuild();
    const double t1 = omp_get_wtime();
    t.pairs = t1 - t0;

    std::unique_ptr<SequentialOracle> oracle;
    if (cfg.verify_batches) oracle = std::make_unique<SequentialOracle>(table, cfg);

    // Phase 3.
    tok_mark_.assign(token_bytes_.size(), 0);
    tok_slot_.assign(token_bytes_.size(), 0);
    Batch batch;
    PairCounts delta;  // serial rounds
    std::vector<ThreadDelta> thread_delta(par ? cfg.threads : 0);
    std::vector<ShardDelta> shard_delta(counts.n_shards());
    std::vector<PairKey> zero;  // pairs at 0 still in the owners' indexes
    for (uint32_t round = 0; merges.size() < n_merges; ++round) {
      const double t_select = omp_get_wtime();
      // One merge is always exact, so a batch is only worth its selection cost if it can go
      // parallel. Without threads, no round can.
      const auto worth_batching = [&](std::span<const Cand> members, size_t share) {
        return !cfg.batch_skip_small ||
               reaches_threshold(table, owners, members, cfg.par_threshold_tokens / share);
      };
      const size_t left = n_merges - merges.size();
      const size_t budget = cfg.batch_skip_small && !par ? 1 : left;
      const size_t k = select_round(heap, budget, cfg.batch_scan_cap, worth_batching);
      stats_.select_s += omp_get_wtime() - t_select;
      if (k == 0) break;  // nothing left to merge
      record_round(k);

      if (oracle) check_round(*oracle, round, merges.size(), k);

      // Before any push: the heap comparator reads the bytes of pairs that hold the new tokens.
      const auto z_first = static_cast<TokenId>(token_bytes_.size());
      for (size_t i = 0; i < k; ++i) {
        const TokenId a = pair_first(prefix_[i].pair), b = pair_second(prefix_[i].pair);
        token_bytes_.push_back(token_bytes_[a] + token_bytes_[b]);
        merges.push_back({a, b, z_first + static_cast<TokenId>(i)});
      }
      tok_mark_.resize(token_bytes_.size(), 0);
      tok_slot_.resize(token_bytes_.size(), 0);
      batch.assign(std::span(prefix_).first(k), z_first, token_bytes_.size(), round);

      double t_order = 0;
      if (par &&
          reaches_threshold(table, owners, std::span(prefix_).first(k), cfg.par_threshold_tokens)) {
        ++stats_.parallel_rounds;
        parallel_step(
            table, owners, counts,
            [&batch](WordTable& tb, Owner& o, auto&& add) { merge_owner_batch(tb, o, batch, add); },
            thread_delta, shard_delta, cfg.threads, /*erase_zero_pairs=*/false);
        t_order = omp_get_wtime();
        for (const ShardDelta& sd : shard_delta) {
          if (sd.negative) throw std::logic_error("BatchEngine: negative pair count");
          stats_.changed_pairs += sd.push.size() + sd.zero.size();
          for (const HeapEntry& e : sd.push) heap.push(e.count, e.pair);
          zero.insert(zero.end(), sd.zero.begin(), sd.zero.end());
        }
      } else {
        delta.clear();
        const auto add = [&delta](PairKey key, int64_t d) { delta[key] += d; };
        for (Owner& owner : owners) merge_owner_batch(table, owner, batch, add);
        t_order = omp_get_wtime();
        for (const auto& [key, d] : delta) {
          if (d == 0) continue;
          ++stats_.changed_pairs;
          const int64_t c = counts.add(key, d);
          if (c > 0) {
            heap.push(static_cast<uint64_t>(c), key);
          } else if (c == 0) {
            counts.erase(key);
            zero.push_back(key);
          } else {
            throw std::logic_error("BatchEngine: negative pair count");
          }
        }
      }
      if (heap.size() > 4 * counts.size()) heap.rebuild();
      stats_.order_s += omp_get_wtime() - t_order;
      if (zero.size() >= kZeroSweepPairs) sweep_zero_pairs(owners, zero, cfg.threads);
    }
    if (oracle && merges.size() < n_merges) check_exhausted(*oracle);
    t.merge = omp_get_wtime() - t1;
    return merges;
  }

  MergeStats stats() const override { return stats_; }

 private:
  static constexpr size_t kZeroSweepPairs = 1 << 14;
  // A round is cut to one merge at once if its best pair has under 1/share of the threshold: only
  // the merge rate suffers on the borderline, because one merge is always exact.
  static constexpr size_t kFirstMemberShare = 8;

  // Picks this round's members into prefix_[0..k) and returns k (0: no live pair left). Every
  // other popped entry goes back on the heap before anything is applied.
  template <class Worth>
  size_t select_round(PairHeap& heap, size_t budget, size_t scan_cap, Worth&& worth_batching) {
    prefix_.clear();
    tail_.clear();
    const uint32_t prefix_stamp = ++stamp_, tail_stamp = ++stamp_;

    // The prefix: best-first, pairwise token-disjoint, a self-pair only in last place.
    BatchStop stop = BatchStop::Disjoint;
    Cand c{};
    bool have = false;       // c is popped but not placed yet
    bool exhausted = false;  // the heap has no live entry left
    for (;;) {
      if (prefix_.size() == budget) {
        stop = BatchStop::Budget;
        break;
      }
      if (!heap.pop_live(c)) {
        exhausted = true;
        break;
      }
      const TokenId a = pair_first(c.pair), b = pair_second(c.pair);
      if (tok_mark_[a] == prefix_stamp || tok_mark_[b] == prefix_stamp) {
        have = true;
        break;
      }
      tok_slot_[a] = tok_slot_[b] = static_cast<uint32_t>(prefix_.size());
      prefix_.push_back(c);
      if (prefix_.size() == 1 && budget > 1 &&
          !worth_batching(std::span<const Cand>(prefix_), kFirstMemberShare)) {
        ++stats_.small_rounds;
        return 1;  // not worth popping the rest of a prefix
      }
      tok_mark_[a] = tok_mark_[b] = prefix_stamp;
      if (a == b) {
        stop = BatchStop::Self;
        break;
      }
    }
    if (prefix_.empty()) return 0;

    // The tie group T, starting with the entry that stopped the prefix.
    const size_t K = prefix_.size();
    if (K > 1 && !worth_batching(std::span<const Cand>(prefix_), 1)) {
      ++stats_.small_rounds;
      if (have) heap.push(c.count, c.pair);
      for (size_t i = 1; i < K; ++i) heap.push(prefix_[i].count, prefix_[i].pair);
      return 1;
    }
    const uint64_t c_star = prefix_.back().count;
    size_t g = 0;
    for (const Cand& m : prefix_) g += m.count > c_star;
    const size_t floor_k = std::max<size_t>(g, 1);
    size_t j = K;  // first member whose tokens appear in the scanned part of T
    bool complete = true;
    while (!exhausted && keep_count(rule_, K, j, floor_k, tail_.empty()) > floor_k) {
      if (!have && !heap.pop_live(c)) break;
      have = true;
      if (c.count < c_star) break;
      if (tail_.size() == scan_cap) {
        complete = false;
        break;
      }
      tail_.push_back(c);
      have = false;
      for (const TokenId tok : {pair_first(c.pair), pair_second(c.pair)}) {
        if (tok_mark_[tok] == prefix_stamp) j = std::min<size_t>(j, tok_slot_[tok]);
      }
    }

    const BatchChoice choice =
        choose_batch(prefix_, tail_, complete, tok_mark_, tail_stamp, rule_, stop);
    ++stats_.stops[static_cast<size_t>(choice.stop)];

    if (have) heap.push(c.count, c.pair);
    for (const Cand& e : tail_) heap.push(e.count, e.pair);
    for (size_t i = choice.k; i < prefix_.size(); ++i) heap.push(prefix_[i].count, prefix_[i].pair);
    return choice.k;
  }

  void record_round(size_t k) {
    ++stats_.rounds;
    stats_.max_batch = std::max<uint64_t>(stats_.max_batch, k);
    size_t bucket = 0;
    while (bucket + 1 < stats_.batch_hist.size() && k > (size_t{1} << bucket)) ++bucket;
    ++stats_.batch_hist[bucket];
  }

  std::string pair_text(uint64_t count, PairKey pair) const {
    return "(" + std::to_string(pair_first(pair)) + "," + std::to_string(pair_second(pair)) +
           ") count " + std::to_string(count);
  }

  void check_round(SequentialOracle& oracle, uint32_t round, size_t merges_done, size_t k) const {
    for (size_t i = 0; i < k; ++i) {
      Cand next{};
      const bool found = oracle.next(next);
      if (found && next.pair == prefix_[i].pair && next.count == prefix_[i].count) continue;
      throw std::logic_error("verify-batches: round " + std::to_string(round) + " slot " +
                             std::to_string(i) + " of " + std::to_string(k) + " (merge " +
                             std::to_string(merges_done + i) + "): batch has " +
                             pair_text(prefix_[i].count, prefix_[i].pair) + ", sequential has " +
                             (found ? pair_text(next.count, next.pair) : "no pair left"));
    }
  }

  void check_exhausted(SequentialOracle& oracle) const {
    Cand next{};
    if (oracle.next(next)) {
      throw std::logic_error("verify-batches: batch ran out of pairs, sequential still has " +
                             pair_text(next.count, next.pair));
    }
  }

  BatchRule rule_;
  MergeStats stats_;
  std::vector<std::string> token_bytes_;
  std::vector<Cand> prefix_, tail_;
  std::vector<uint32_t> tok_mark_,
      tok_slot_;  // tok_slot_: valid where tok_mark_ has the prefix stamp
  uint32_t stamp_ = 0;
};

}  // namespace

std::unique_ptr<MergeEngine> make_batch_engine(BatchRule rule) {
  return std::make_unique<BatchEngine>(rule);
}

std::unique_ptr<MergeEngine> make_batch_engine() {
  return make_batch_engine(BatchRule::Closed);
}

}  // namespace parabpe::detail
