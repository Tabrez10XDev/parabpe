// Batched merging (src/batch_select.hpp, src/merge_batch.cpp): the selection rule, the one-pass
// apply, and the engine against NaiveSerial with the sequential oracle switched on.

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "batch_select.hpp"
#include "bpe_common.hpp"
#include "incr_common.hpp"
#include "parabpe/merge_engine.hpp"
#include "parabpe/trainer.hpp"
#include "strategies.hpp"

using namespace parabpe;
using detail::BatchChoice;
using detail::BatchRule;
using detail::Cand;
using detail::choose_batch;

namespace {

Cand cand(uint64_t count, TokenId a, TokenId b) {
  return {count, make_pair_key(a, b)};
}

BatchChoice choose(const std::vector<Cand>& prefix, const std::vector<Cand>& tail,
                   BatchRule rule = BatchRule::Closed, bool complete = true) {
  std::vector<uint32_t> tok_mark(64, 0);
  return choose_batch(prefix, tail, complete, tok_mark, 1, rule);
}

std::string docs(const std::vector<std::pair<std::string, int>>& words) {
  std::string text;
  for (const auto& [word, n] : words) {
    for (int i = 0; i < n; ++i) text += word + "<|endoftext|>";
  }
  return text;
}

std::vector<Merge> naive_merges(const std::string& text, size_t vocab_size) {
  TrainConfig cfg;
  cfg.vocab_size = vocab_size;
  return train_text(text, cfg).merges;
}

// The batched result with every member checked by the sequential oracle.
TrainResult verified(const std::string& text, size_t vocab_size, size_t scan_cap = 256) {
  TrainConfig cfg;
  cfg.vocab_size = vocab_size;
  cfg.merge = MergeStrategy::IncrBatch;
  cfg.verify_batches = true;
  cfg.batch_scan_cap = scan_cap;
  cfg.batch_skip_small = false;
  return train_text(text, cfg);
}

void expect_exact(const std::string& text, size_t vocab_size) {
  const std::vector<Merge> naive = naive_merges(text, vocab_size);
  for (size_t cap : {size_t{0}, size_t{1}, size_t{2}, size_t{256}}) {
    EXPECT_EQ(verified(text, vocab_size, cap).merges, naive) << "cap " << cap;
  }
}

}  // namespace

TEST(ChooseBatch, TieWithFirstMemberKeepsOnlyIt) {
  // From the note's counterexample: T touches p[0], so j = 0 and only p[0] is safe.
  const BatchChoice c = choose({cand(5, 1, 2), cand(5, 3, 4)}, {cand(5, 2, 9)});
  EXPECT_EQ(c.k, 1u);
  EXPECT_EQ(c.stop, BatchStop::Tie);
}

TEST(ChooseBatch, TieOnlyWithLastMemberKeepsEverything) {
  const std::vector<Cand> prefix = {cand(5, 1, 2), cand(5, 3, 4)}, tail = {cand(5, 4, 9)};
  EXPECT_EQ(choose(prefix, tail).k, 2u);                      // the last member's tokens are free
  EXPECT_EQ(choose(prefix, tail, BatchRule::Relaxed).k, 1u);  // the relaxed rule checks them
  EXPECT_EQ(choose(prefix, tail, BatchRule::Strict).k, 1u);
}

TEST(ChooseBatch, MembersAboveTheTieCountAlwaysStay) {
  // T touches p[0], but p[0] and p[1] beat every outside pair on count.
  EXPECT_EQ(choose({cand(9, 1, 2), cand(7, 3, 4), cand(5, 5, 6)}, {cand(5, 1, 9)}).k, 2u);
}

TEST(ChooseBatch, TiedMemberAfterTheFirstConflictIsDropped) {
  EXPECT_EQ(choose({cand(9, 1, 2), cand(5, 3, 4), cand(5, 5, 6), cand(5, 7, 8)}, {cand(5, 3, 9)}).k,
            2u);
}

TEST(ChooseBatch, EmptyOrUnrelatedTail) {
  const std::vector<Cand> prefix = {cand(5, 1, 2), cand(5, 3, 4)};
  for (BatchRule rule : {BatchRule::Strict, BatchRule::Relaxed, BatchRule::Closed}) {
    EXPECT_EQ(choose(prefix, {}, rule).k, 2u);
  }
  EXPECT_EQ(choose(prefix, {cand(5, 7, 8)}).k, 2u);
  EXPECT_EQ(choose(prefix, {cand(5, 7, 8)}, BatchRule::Relaxed).k, 2u);
  EXPECT_EQ(choose(prefix, {cand(5, 7, 8)}, BatchRule::Strict).k, 1u);
}

TEST(ChooseBatch, IncompleteScanFallsBackToMembersAboveTheTieCount) {
  const BatchChoice c = choose({cand(5, 1, 2), cand(5, 3, 4)}, {}, BatchRule::Closed, false);
  EXPECT_EQ(c.k, 1u);
  EXPECT_EQ(c.stop, BatchStop::Capped);
  EXPECT_EQ(choose({cand(9, 1, 2), cand(8, 3, 4), cand(5, 5, 6)}, {}, BatchRule::Closed, false).k,
            2u);
}

TEST(ChooseBatch, SingleMemberAndStopReasonPassThrough) {
  std::vector<uint32_t> tok_mark(64, 0);
  const std::vector<Cand> one = {cand(5, 1, 2)}, tail = {cand(5, 1, 3)};
  const BatchChoice c =
      choose_batch(one, tail, true, tok_mark, 1, BatchRule::Closed, BatchStop::Budget);
  EXPECT_EQ(c.k, 1u);
  EXPECT_EQ(c.stop, BatchStop::Budget);
}

TEST(ApplyBatch, MatchesSequentialMerges) {
  std::mt19937 rng(777);
  for (int trial = 0; trial < 5000; ++trial) {
    // Random disjoint members over tokens 0..7; a self-pair can land in any slot.
    std::vector<TokenId> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    std::shuffle(tokens.begin(), tokens.end(), rng);
    std::vector<Cand> members;
    size_t next = 0;
    const size_t k = 1 + rng() % 3;
    for (size_t i = 0; i < k; ++i) {
      if (rng() % 3 == 0) {
        members.push_back(cand(1, tokens[next], tokens[next]));
        next += 1;
      } else {
        members.push_back(cand(1, tokens[next], tokens[next + 1]));
        next += 2;
      }
    }
    const TokenId z_first = 100;
    detail::Batch batch;
    batch.assign(members, z_first, z_first + k, static_cast<uint32_t>(trial));

    std::vector<TokenId> word(rng() % 24);
    for (TokenId& t : word) t = tokens[rng() % 5];
    std::vector<TokenId> expected = word;
    uint32_t len = static_cast<uint32_t>(expected.size());
    for (size_t i = 0; i < k; ++i) {
      len = detail::apply_merge(expected.data(), len, pair_first(members[i].pair),
                                pair_second(members[i].pair), z_first + static_cast<TokenId>(i));
    }
    expected.resize(len);
    EXPECT_EQ(detail::contains_member(word.data(), static_cast<uint32_t>(word.size()), batch),
              expected.size() != word.size());
    word.resize(detail::apply_batch(word.data(), static_cast<uint32_t>(word.size()), batch));
    ASSERT_EQ(word, expected) << "trial " << trial;
  }
}

TEST(ApplyBatch, AssignClearsThePreviousRound) {
  detail::Batch batch;
  batch.assign(std::vector<Cand>{cand(1, 'a', 'b')}, 300, 301, 0);
  batch.assign(std::vector<Cand>{cand(1, 'c', 'd')}, 301, 302, 1);
  EXPECT_FALSE(batch.matches('a', 'b'));
  EXPECT_TRUE(batch.matches('c', 'd'));
}

TEST(BatchEngine, NewPairBeatsATiedMember) {
  // The note's counterexample: after (e,a), the new pair (ea,x) beats (d,f) on the tie-break, and
  // (a,x), which stopped the prefix, is what shows it.
  const std::string text = docs({{"eax", 2}, {"df", 2}});
  expect_exact(text, 256 + 1 + 3);
  const TrainResult r = verified(text, 256 + 1 + 3);
  EXPECT_EQ(r.merges[0], (Merge{'e', 'a', 257}));
  EXPECT_EQ(r.merges[1], (Merge{257, 'x', 258}));
  EXPECT_EQ(r.merges[2], (Merge{'d', 'f', 259}));
}

TEST(BatchEngine, SelfPairNeverRunsMidBatch) {
  // (a,a) has count 200; (aa,a) then has 100 and must go before (b,c) with 50.
  expect_exact(docs({{"aaa", 100}, {"bc", 50}}), 256 + 1 + 3);
}

TEST(BatchEngine, DisjointPairsLandInOneRound) {
  const TrainResult r = verified(docs({{"ab", 9}, {"cd", 7}, {"ef", 5}}), 256 + 1 + 3);
  EXPECT_EQ(r.stats.rounds, 1u);
  EXPECT_EQ(r.stats.max_batch, 3u);
  EXPECT_EQ(r.merges, naive_merges(docs({{"ab", 9}, {"cd", 7}, {"ef", 5}}), 256 + 1 + 3));
}

TEST(BatchEngine, BudgetCutsTheRoundShort) {
  const std::string text = docs({{"ab", 9}, {"cd", 7}, {"ef", 5}, {"gh", 3}});
  const TrainResult r = verified(text, 256 + 1 + 2);
  EXPECT_EQ(r.merges, naive_merges(text, 256 + 1 + 2));
  EXPECT_EQ(r.stats.stops[static_cast<size_t>(BatchStop::Budget)], 1u);
  expect_exact(text, 256 + 1 + 1);
}

TEST(BatchEngine, AdversarialCorpora) {
  // Long alphabet word: everything ties at 1 and chains through its neighbours.
  expect_exact("abcdefghijklmnopqrstuvwxyz", 256 + 1 + 25);
  // Left-neighbour ties: (x,a) ties with (a,b), and (x,ab) can then beat other tied pairs.
  expect_exact(docs({{"xab", 3}, {"cd", 3}, {"yab", 3}, {"ef", 3}}), 256 + 1 + 12);
  expect_exact(docs({{"xaby", 4}, {"xa", 2}, {"by", 2}, {"qr", 6}, {"st", 6}}), 256 + 1 + 12);
  // Runs of a single letter, all lengths at once.
  std::vector<std::pair<std::string, int>> runs;
  for (int n = 2; n < 20; ++n) runs.push_back({std::string(static_cast<size_t>(n), 'a'), 1});
  expect_exact(docs(runs), 256 + 1 + 40);
  // Small alphabets and uniform frequencies.
  expect_exact(std::string(257, 'a'), 256 + 1 + 40);
  expect_exact(docs({{"abababab", 3}, {"babababa", 3}, {"aabb", 3}, {"bbaa", 3}}), 256 + 1 + 40);
  std::vector<std::pair<std::string, int>> uniform;
  for (char a = 'a'; a <= 'h'; ++a) {
    for (char b = 'a'; b <= 'h'; ++b) uniform.push_back({std::string{a, b}, 1});
  }
  expect_exact(docs(uniform), 256 + 1 + 30);
}

TEST(BatchEngine, EveryRuleIsExactOnRandomCorpora) {
  const auto run_rule = [](BatchRule rule, const std::string& text, const TrainConfig& cfg) {
    PhaseTimings t;
    WordTable table = build_word_table(count_words(text, cfg));
    return detail::make_batch_engine(rule)->run(table, cfg, t);
  };
  std::mt19937 rng(4242);
  for (int trial = 0; trial < 200; ++trial) {
    std::string text;
    const int n = 1 + static_cast<int>(rng() % 400);
    const std::string alphabet = trial % 3 == 0 ? "ab " : trial % 3 == 1 ? "abc \n" : "abcdefg \n";
    for (int i = 0; i < n; ++i) text += alphabet[rng() % alphabet.size()];
    TrainConfig cfg;
    cfg.vocab_size = 256 + 1 + 1 + rng() % 120;
    const std::vector<Merge> naive = train_text(text, cfg).merges;
    cfg.verify_batches = true;
    cfg.batch_skip_small = false;
    cfg.batch_scan_cap = trial % 4 == 0 ? rng() % 4 : 256;
    for (BatchRule rule : {BatchRule::Strict, BatchRule::Relaxed, BatchRule::Closed}) {
      ASSERT_EQ(run_rule(rule, text, cfg), naive)
          << "rule " << static_cast<int>(rule) << " trial " << trial << " text \"" << text << "\"";
    }
  }
}

TEST(HashedOwners, EveryWordOnceInIncreasingOrder) {
  WordTable t;
  t.offset.resize(10000);
  for (const size_t n : {size_t{1}, size_t{3}, size_t{20}, size_t{44}}) {
    const std::vector<detail::Owner> owners = detail::partition_words_hashed(t, n);
    ASSERT_EQ(owners.size(), n);
    std::vector<int> seen(t.size(), 0);
    for (const detail::Owner& o : owners) {
      EXPECT_TRUE(std::is_sorted(o.words.begin(), o.words.end()));
      o.for_each_word([&seen](uint32_t w) { ++seen[w]; });
    }
    EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; }));
  }
}

TEST(BatchEngine, ParallelRoundsMatchNaive) {
  std::mt19937 rng(9001);
  for (int trial = 0; trial < 60; ++trial) {
    std::string text;
    const int n = 200 + static_cast<int>(rng() % 3000);
    const std::string alphabet = trial % 2 ? "abcd \n" : "abcdefghij  \n";
    for (int i = 0; i < n; ++i) text += alphabet[rng() % alphabet.size()];
    TrainConfig cfg;
    cfg.vocab_size = 256 + 1 + 1 + rng() % 200;
    const std::vector<Merge> naive = train_text(text, cfg).merges;
    cfg.merge = MergeStrategy::IncrBatch;
    cfg.verify_batches = true;
    for (const int threads : {2, 3, 8, 16}) {
      for (const size_t owners : {size_t{1}, size_t{4}}) {
        for (const size_t threshold : {size_t{0}, size_t{32}}) {
          cfg.threads = threads;
          cfg.batch_skip_small = trial % 2 == 0;
          cfg.owners_per_thread = owners;
          cfg.par_threshold_tokens = threshold;
          const TrainResult r = train_text(text, cfg);
          ASSERT_EQ(r.merges, naive) << "threads " << threads << " owners " << owners
                                     << " threshold " << threshold << " trial " << trial;
          if (threshold == 0 && !r.merges.empty()) {
            EXPECT_EQ(r.stats.parallel_rounds, r.stats.rounds);
          }
        }
      }
    }
  }
}

TEST(OwnerMap, HashedIncrParMatchesNaive) {
  std::mt19937 rng(4242);
  for (int trial = 0; trial < 60; ++trial) {
    std::string text;
    const int n = 200 + static_cast<int>(rng() % 3000);
    const std::string alphabet = trial % 2 ? "abcd \n" : "abcdefghij  \n";
    for (int i = 0; i < n; ++i) text += alphabet[rng() % alphabet.size()];
    TrainConfig cfg;
    cfg.vocab_size = 256 + 1 + 1 + rng() % 200;
    const std::vector<Merge> naive = train_text(text, cfg).merges;
    cfg.merge = MergeStrategy::IncrPar;
    cfg.owner_map = OwnerMap::Hashed;
    cfg.owner_stats = trial % 3 == 0;  // measuring must not change the result
    for (const int threads : {1, 2, 3, 8, 16}) {
      for (const size_t threshold : {size_t{0}, size_t{32}}) {
        cfg.threads = threads;
        cfg.par_threshold_tokens = threshold;
        ASSERT_EQ(train_text(text, cfg).merges, naive)
            << "threads " << threads << " threshold " << threshold << " trial " << trial;
      }
    }
  }
}

// 1000 words starting "qu" and 1000 starting "xy": sorted by bytes, each group is one half of the
// table, so the two big merges each land entirely in one of two contiguous owners.
TEST(OwnerMap, ContiguousClustersAPairAndHashedSpreadsIt) {
  std::vector<std::pair<std::string, int>> words;
  for (const std::string prefix : {"qu", "xy"}) {
    for (int i = 0; i < 1000; ++i) {
      std::string w = prefix;
      for (int d = 0, v = i; d < 4; ++d, v /= 8) w += static_cast<char>('a' + v % 8);
      words.push_back({w, 1});
    }
  }
  const std::string text = docs(words);
  TrainConfig cfg;
  cfg.vocab_size = 256 + 1 + 6;  // the two big merges, then four small ones that must not count
  cfg.merge = MergeStrategy::IncrPar;
  cfg.threads = 2;
  EXPECT_EQ(train_text(text, cfg).stats.big_steps, 0u);  // off by default

  cfg.owner_stats = true;
  const TrainResult contiguous = train_text(text, cfg);
  ASSERT_EQ(contiguous.stats.big_steps, 2u);
  EXPECT_DOUBLE_EQ(contiguous.stats.owner_busiest / contiguous.stats.owner_mean, 2.0);
  EXPECT_DOUBLE_EQ(contiguous.stats.owner_mean, 2.0 * 6000 / 2);

  cfg.owner_map = OwnerMap::Hashed;
  const TrainResult hashed = train_text(text, cfg);
  ASSERT_EQ(hashed.stats.big_steps, 2u);
  EXPECT_DOUBLE_EQ(hashed.stats.owner_mean, contiguous.stats.owner_mean);
  EXPECT_LT(hashed.stats.owner_busiest / hashed.stats.owner_mean, 1.3);
  EXPECT_EQ(hashed.merges, contiguous.merges);
}

TEST(BatchEngine, ZeroPairSweepKeepsResultsExact) {
  // Enough distinct pairs that many reach 0 and the periodic sweep runs mid-training.
  std::mt19937 rng(31337);
  std::string text;
  for (int doc = 0; doc < 3000; ++doc) {
    const int len = 2 + static_cast<int>(rng() % 12);
    for (int i = 0; i < len; ++i) text += static_cast<char>('a' + rng() % 26);
    text += "<|endoftext|>";
  }
  TrainConfig cfg;
  cfg.vocab_size = 256 + 1 + 6000;
  const std::vector<Merge> naive = train_text(text, cfg).merges;
  cfg.merge = MergeStrategy::IncrBatch;
  cfg.verify_batches = true;
  cfg.batch_skip_small = false;
  for (const int threads : {1, 3, 8}) {
    cfg.threads = threads;
    cfg.par_threshold_tokens = 0;
    EXPECT_EQ(train_text(text, cfg).merges, naive) << "threads " << threads;
  }
}

TEST(BatchEngine, SmallRoundsAreSkippedWithoutChangingTheResult) {
  const std::string text = docs({{"ab", 9}, {"cd", 7}, {"ef", 5}, {"gh", 3}});
  TrainConfig cfg;
  cfg.vocab_size = 256 + 1 + 4;
  const std::vector<Merge> naive = train_text(text, cfg).merges;
  cfg.merge = MergeStrategy::IncrBatch;
  cfg.verify_batches = true;
  cfg.threads = 3;
  cfg.par_threshold_tokens = 1 << 20;  // no round can reach it
  const TrainResult skipped = train_text(text, cfg);
  EXPECT_EQ(skipped.merges, naive);
  EXPECT_EQ(skipped.stats.rounds, 4u);
  EXPECT_GT(skipped.stats.small_rounds, 0u);
  cfg.batch_skip_small = false;
  const TrainResult full = train_text(text, cfg);
  EXPECT_EQ(full.merges, naive);
  EXPECT_EQ(full.stats.rounds, 1u);
  EXPECT_EQ(full.stats.small_rounds, 0u);
  // Without threads no round can go parallel, so every round is a single merge.
  cfg.batch_skip_small = true;
  cfg.threads = 1;
  EXPECT_EQ(train_text(text, cfg).stats.rounds, 4u);
}

TEST(BatchEngine, StatsAddUp) {
  const TrainResult r = verified(docs({{"abcab", 5}, {"cdcd", 3}, {"efg", 2}}), 256 + 1 + 8);
  uint64_t rounds = 0;
  for (const uint64_t n : r.stats.batch_hist) rounds += n;
  EXPECT_EQ(rounds, r.stats.rounds);
  uint64_t stops = 0;
  for (const uint64_t n : r.stats.stops) stops += n;
  EXPECT_EQ(stops + r.stats.small_rounds, r.stats.rounds);
  EXPECT_GT(r.stats.changed_pairs, 0u);
}
