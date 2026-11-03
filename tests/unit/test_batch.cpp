// Batched merging (src/batch_select.hpp, src/merge_batch.cpp): the selection rule, the one-pass
// apply, and the engine against NaiveSerial with the sequential oracle switched on.

#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "batch_select.hpp"
#include "bpe_common.hpp"
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
    cfg.batch_scan_cap = trial % 4 == 0 ? rng() % 4 : 256;
    for (BatchRule rule : {BatchRule::Strict, BatchRule::Relaxed, BatchRule::Closed}) {
      ASSERT_EQ(run_rule(rule, text, cfg), naive)
          << "rule " << static_cast<int>(rule) << " trial " << trial << " text \"" << text << "\"";
    }
  }
}

TEST(BatchEngine, StatsAddUp) {
  const TrainResult r = verified(docs({{"abcab", 5}, {"cdcd", 3}, {"efg", 2}}), 256 + 1 + 8);
  uint64_t rounds = 0;
  for (const uint64_t n : r.stats.batch_hist) rounds += n;
  EXPECT_EQ(rounds, r.stats.rounds);
  uint64_t stops = 0;
  for (const uint64_t n : r.stats.stops) stops += n;
  EXPECT_EQ(stops, r.stats.rounds);
  EXPECT_GT(r.stats.changed_pairs, 0u);
}
