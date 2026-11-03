// Writers, vocab, word table and the shared merge rules (src/bpe_common.hpp).

#include <gtest/gtest.h>

#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "bpe_common.hpp"
#include "parabpe/io.hpp"
#include "parabpe/merge_engine.hpp"
#include "parabpe/trainer.hpp"

using namespace parabpe;

namespace {

constexpr TokenId kA = 'a', kB = 'b', kZ = 'z';

std::vector<TokenId> merged(std::vector<TokenId> w, TokenId a, TokenId b, TokenId z) {
  w.resize(detail::apply_merge(w.data(), static_cast<uint32_t>(w.size()), a, b, z));
  return w;
}

}  // namespace

TEST(BytesToUnicode, Gpt2Table) {
  const auto& t = bytes_to_unicode();
  EXPECT_EQ(t[' '], "\xC4\xA0");   // Ġ (U+0120)
  EXPECT_EQ(t['\n'], "\xC4\x8A");  // Ċ (U+010A)
  EXPECT_EQ(t['a'], "a");
  EXPECT_EQ(t[0], "\xC4\x80");     // U+0100
  EXPECT_EQ(t[0xAD], "\xC5\x83");  // soft hyphen is not printable: U+0143
  EXPECT_EQ(t[0xFF], "\xC3\xBF");  // ÿ maps to itself
  EXPECT_EQ(std::set<std::string>(t.begin(), t.end()).size(), 256u);
}

TEST(JsonString, EscapesLikePython) {
  // json.dumps("\x01\"\\\n\t\x7f é ", ensure_ascii=False)
  EXPECT_EQ(json_string("\x01\"\\\n\t\x7f \xC3\xA9\xE2\x80\xA8"),
            "\"\\u0001\\\"\\\\\\n\\t\x7f \xC3\xA9\xE2\x80\xA8\"");
  EXPECT_EQ(json_string("\b\f\r\x1f"), "\"\\b\\f\\r\\u001f\"");
}

TEST(Vocab, BuildAndFormat) {
  const std::vector<Merge> merges = {{'a', 'b', 257}, {' ', 257, 258}};
  const Vocab v = build_vocab(merges, {"<|endoftext|>"});
  ASSERT_EQ(v.size(), 259u);
  EXPECT_EQ(v[256], "<|endoftext|>");
  EXPECT_EQ(v[258], " ab");
  EXPECT_EQ(format_merges(merges, v), "#version: 0.2\na b\n\xC4\xA0 ab\n");
  const std::string json = format_vocab(v, 1);
  EXPECT_TRUE(json.starts_with("{\"\xC4\x80\":0,\"\xC4\x81\":1,")) << json.substr(0, 20);
  EXPECT_TRUE(
      json.ends_with(",\"<|endoftext|>\":256,\"ab\":257,\"\xC4\xA0"
                     "ab\":258}"));
}

TEST(Vocab, RejectsOutOfOrderMergesAndDuplicateKeys) {
  EXPECT_THROW(build_vocab({{'a', 'b', 300}}, {}), std::invalid_argument);
  // A special token "a" collides with byte token "a" in vocab.json (Python asserts too).
  EXPECT_THROW(format_vocab(build_vocab({}, {"a"}), 1), std::runtime_error);
}

TEST(Timings, ContainsSchemaFields) {
  TrainResult r;
  r.merges = {{'a', 'b', 257}};
  TrainConfig cfg;
  const std::string json = format_timings(r, "data/x \"quoted\".txt", cfg);
  for (const char* key :
       {"\"merges\": 1", "\"chunk\"", "\"pretok\"", "\"build\"", "\"pairs\"", "\"merge\"",
        "\"total\"", "\"input\": \"data/x \\\"quoted\\\".txt\"", "\"vocab_size\": 32000",
        "\"pretok\": \"serial\"", "\"pretok_reduce\": \"bucketed\"", "\"merge\": \"incr-serial\"",
        "\"cpu_model\"", "\"physical_cores\"", "\"logical_cores\"", "\"smt\"", "\"os\"",
        "\"git_sha\""}) {
    EXPECT_NE(json.find(key), std::string::npos) << key;
  }
#ifdef __APPLE__
  EXPECT_NE(json.find("\"performance_cores\""), std::string::npos);
#endif
}

TEST(Timings, BatchBlockOnlyForIncrBatch) {
  TrainResult r;
  r.stats.rounds = 7;
  r.stats.batch_hist[2] = 3;
  r.stats.stops[static_cast<size_t>(BatchStop::Capped)] = 2;
  TrainConfig cfg;
  EXPECT_EQ(format_timings(r, "x", cfg).find("\"batch\""), std::string::npos);
  cfg.merge = MergeStrategy::IncrBatch;
  const std::string json = format_timings(r, "x", cfg);
  for (const char* key : {"\"merge\": \"incr-batch\"", "\"batch\": {\"rounds\": 7",
                          "\"3-4\": 3", "\"capped\": 2", "\"scan_cap\": 4096"}) {
    EXPECT_NE(json.find(key), std::string::npos) << key;
  }
}

TEST(MergeRules, GreedyLeftToRightNoOverlap) {
  EXPECT_EQ(merged({kA, kA, kA, kA}, kA, kA, 256), (std::vector<TokenId>{256, 256}));
  EXPECT_EQ(merged({kA, kA, kA}, kA, kA, 256), (std::vector<TokenId>{256, kA}));
  EXPECT_EQ(merged({kA, kB, kA, kB, kZ}, kA, kB, 256), (std::vector<TokenId>{256, 256, kZ}));
  EXPECT_EQ(merged({kB}, kA, kB, 256), (std::vector<TokenId>{kB}));
  EXPECT_EQ(merged({}, kA, kB, 256), (std::vector<TokenId>{}));
}

TEST(MergeRules, TieBreakComparesTuplesNotJoinedBytes) {
  TrainConfig cfg;
  cfg.special_tokens = {};
  auto tb = detail::initial_token_bytes(cfg);
  tb.push_back("ab");  // id 256
  const PairKey ab_a = make_pair_key(256, kA), a_z = make_pair_key(kA, kZ);
  // Joined bytes would pick (a,z): "az" > "aba". The tuple rule compares "ab" > "a" first.
  EXPECT_TRUE(detail::better_pair(5, ab_a, 5, a_z, tb));
  EXPECT_FALSE(detail::better_pair(5, a_z, 5, ab_a, tb));
  EXPECT_TRUE(detail::better_pair(6, a_z, 5, ab_a, tb));  // count first
  EXPECT_TRUE(detail::better_pair(3, make_pair_key(kA, kZ), 3, make_pair_key(kA, kA), tb));
  // Bytes compare unsigned: 0xC3 > 'z'.
  EXPECT_TRUE(detail::better_pair(1, make_pair_key(0xC3, kA), 1, make_pair_key(kZ, kA), tb));
}

TEST(MergeRules, NumMergesAndVocabTooSmall) {
  TrainConfig cfg;
  cfg.vocab_size = 300;
  cfg.special_tokens = {"<|a|>", "<|b|>"};
  EXPECT_EQ(detail::num_merges(cfg), 42u);
  cfg.vocab_size = 257;
  EXPECT_THROW(detail::num_merges(cfg), std::invalid_argument);
}

TEST(WordTable, FlattensBytes) {
  WordCounts c;
  c.words = {"ab", "\xC3\xA9", "x"};
  c.freq = {3, 2, 1};
  const WordTable t = build_word_table(c);
  EXPECT_EQ(t.tokens, (std::vector<TokenId>{'a', 'b', 0xC3, 0xA9, 'x'}));
  EXPECT_EQ(t.offset, (std::vector<uint32_t>{0, 2, 4}));
  EXPECT_EQ(t.len, (std::vector<uint32_t>{2, 2, 1}));
  EXPECT_EQ(t.freq, (std::vector<uint64_t>{3, 2, 1}));
  EXPECT_EQ(t.last_touched.size(), 3u);
}

class Engines : public ::testing::TestWithParam<MergeStrategy> {};

TEST_P(Engines, StopsEarlyWhenNoPairsRemain) {
  TrainConfig cfg;
  cfg.vocab_size = 256 + 1 + 50;
  cfg.merge = GetParam();
  const TrainResult r = train_text("ab ab", cfg);
  ASSERT_EQ(r.merges.size(), 2u);  // (a,b) then (Ġ,ab)
  EXPECT_EQ(r.merges[0], (Merge{kA, kB, 257}));
  EXPECT_EQ(r.merges[1], (Merge{' ', 257, 258}));
  EXPECT_EQ(r.vocab.size(), 259u);
}

TEST_P(Engines, EmptyCorpusGivesHeaderOnlyMerges) {
  TrainConfig cfg;
  cfg.vocab_size = 300;
  cfg.merge = GetParam();
  const TrainResult r = train_text("", cfg);
  EXPECT_TRUE(r.merges.empty());
  EXPECT_EQ(format_merges(r.merges, r.vocab), "#version: 0.2\n");
}

INSTANTIATE_TEST_SUITE_P(All, Engines,
                         ::testing::Values(MergeStrategy::NaiveSerial, MergeStrategy::IncrSerial,
                                           MergeStrategy::NaivePar, MergeStrategy::IncrPar,
                                           MergeStrategy::IncrBatch));

// Every other engine against the NaiveSerial oracle on random small-alphabet corpora: long runs,
// repeated pairs inside one word and many count ties stress the index, the stamps and the lazy
// heap (IncrSerial) and the bucketed reduction + max (parallel engines, at several thread counts).
TEST(Engines, MatchNaiveOnRandomCorpora) {
  const std::vector<std::pair<MergeStrategy, int>> engines = {
      {MergeStrategy::IncrSerial, 1}, {MergeStrategy::NaivePar, 1},  {MergeStrategy::NaivePar, 3},
      {MergeStrategy::NaivePar, 4},   {MergeStrategy::NaivePar, 16}, {MergeStrategy::IncrPar, 1},
      {MergeStrategy::IncrPar, 2},    {MergeStrategy::IncrPar, 3},   {MergeStrategy::IncrPar, 8},
      {MergeStrategy::IncrPar, 16},   {MergeStrategy::IncrBatch, 1}, {MergeStrategy::IncrBatch, 2},
      {MergeStrategy::IncrBatch, 3},  {MergeStrategy::IncrBatch, 8}, {MergeStrategy::IncrBatch, 16}};
  std::mt19937 rng(12345);
  for (int trial = 0; trial < 200; ++trial) {
    std::string text;
    const int n = 1 + static_cast<int>(rng() % 400);
    const std::string alphabet = trial % 2 ? "ab " : "abc \n";
    for (int i = 0; i < n; ++i) text += alphabet[rng() % alphabet.size()];
    TrainConfig cfg;
    cfg.vocab_size = 256 + 1 + 1 + rng() % 120;
    cfg.merge = MergeStrategy::NaiveSerial;
    const TrainResult naive = train_text(text, cfg);
    for (const auto& [engine, threads] : engines) {
      cfg.merge = engine;
      cfg.threads = threads;
      cfg.par_threshold_tokens =
          trial % 2 ? 0 : 8;  // IncrPar: all-parallel, or a mix of both paths
      cfg.verify_batches = true;
      const TrainResult other = train_text(text, cfg);
      ASSERT_EQ(other.merges, naive.merges)
          << "engine " << static_cast<int>(engine) << " threads " << threads << " trial " << trial
          << " text \"" << text << "\"";
    }
  }
}
