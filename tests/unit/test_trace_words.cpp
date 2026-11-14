// B10: what Studio's "The work" view reads. PRETOK_UNIT events name their work unit, and
// --trace-words lists frequent words with the phase-3 owner that rewrites them.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "incr_common.hpp"
#include "parabpe/merge_engine.hpp"
#include "parabpe/trainer.hpp"
#include "trace.hpp"
#include "trace_words.hpp"

using namespace parabpe;
using detail::TraceKind;
using detail::Tracer;

namespace {

// Words of 1-9 letters, so plenty reach the 4-byte minimum, with many documents.
std::string word_text(unsigned seed, int words) {
  std::mt19937 rng(seed);
  std::string text;
  for (int i = 0; i < words; ++i) {
    const int len = 1 + static_cast<int>(rng() % 9);
    text += ' ';
    for (int j = 0; j < len; ++j) text += static_cast<char>('a' + rng() % 5);
    if (i % 50 == 49) text += "<|endoftext|>";
  }
  return text;
}

TrainConfig config(int threads, int merge_threads) {
  TrainConfig cfg;
  cfg.vocab_size = 256 + 1 + 150;
  cfg.merge = MergeStrategy::IncrBatch;
  cfg.pretok = PretokStrategy::OMP;
  cfg.threads = threads;
  cfg.merge_threads = merge_threads;
  cfg.owners_per_thread = 1;
  cfg.work_unit_bytes = 512;  // many units
  return cfg;
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string unhex(const std::string& hex) {
  std::string out;
  for (size_t i = 0; i + 1 < hex.size(); i += 2) {
    out += static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16));
  }
  return out;
}

}  // namespace

TEST(TraceWords, PretokUnitsNameTheirWorkUnit) {
  const std::string text = word_text(1, 6000);
  TrainConfig cfg = config(4, 0);
  Tracer tracer(4);
  detail::g_tracer = &tracer;
  train_text(text, cfg);
  detail::g_tracer = nullptr;
  std::vector<uint64_t> units;
  for (int t = 0; t < tracer.threads(); ++t) {
    for (const detail::TraceEvent& e : tracer.events(t)) {
      if (e.kind == TraceKind::PretokUnit) units.push_back(e.arg2);
    }
  }
  ASSERT_GT(units.size(), 8u);
  std::sort(units.begin(), units.end());
  for (size_t i = 0; i < units.size(); ++i) EXPECT_EQ(units[i], i);  // each unit exactly once
  EXPECT_STREQ(detail::trace_arg2_name(TraceKind::PretokUnit), "unit");
  EXPECT_EQ(detail::trace_arg2_name(TraceKind::OwnerMerge), nullptr);

  const std::string path = ::testing::TempDir() + "parabpe_units.json";
  cfg.trace_path = path;
  train_text(text, cfg);
  const std::string json = read_file(path);
  EXPECT_NE(json.find("\"bytes\": "), std::string::npos);
  EXPECT_NE(json.find(", \"unit\": "), std::string::npos);
  std::remove(path.c_str());
}

TEST(TraceWords, SampleIsTheMostFrequentLongWordsWithTheirOwners) {
  const std::string text = word_text(2, 6000);
  const TrainConfig cfg = config(2, 0);
  const WordCounts counts = count_words(text, cfg);
  const WordTable table = build_word_table(counts);
  const auto owners = detail::partition_words_hashed(table, detail::batch_owner_count(cfg));
  const auto sample = detail::trace_word_sample(counts, MergeStrategy::IncrBatch, cfg, 50);
  ASSERT_EQ(sample.size(), 50u);
  uint64_t longer_seen = 0;  // the most frequent long word not in the sample
  std::set<uint32_t> in_sample;
  for (size_t i = 0; i < sample.size(); ++i) {
    const auto& s = sample[i];
    in_sample.insert(s.w);
    EXPECT_GE(counts.words[s.w].size(), detail::kTraceWordMinBytes);
    EXPECT_EQ(s.count, counts.freq[s.w]);
    if (i) EXPECT_GE(sample[i - 1].count, s.count);
    ASSERT_GE(s.owner, 0);
    const auto& mine = owners[static_cast<size_t>(s.owner)].words;
    EXPECT_TRUE(std::binary_search(mine.begin(), mine.end(), s.w)) << "word " << s.w;
  }
  for (uint32_t w = 0; w < counts.words.size(); ++w) {
    if (counts.words[w].size() >= detail::kTraceWordMinBytes && !in_sample.count(w)) {
      longer_seen = std::max(longer_seen, counts.freq[w]);
    }
  }
  EXPECT_LE(longer_seen, sample.back().count);
  for (const auto& s : detail::trace_word_sample(counts, MergeStrategy::IncrPar, cfg, 5)) {
    EXPECT_EQ(s.owner, -1);  // only IncrBatch's owners are known
  }
}

// The owners in the file are phase 3's: 8 threads for pre-tokenizing, 2 for merging.
TEST(TraceWords, FileUsesTheMergeTeamsOwners) {
  const std::string text = word_text(3, 6000);
  const std::string path = ::testing::TempDir() + "parabpe_words.json";
  std::remove(path.c_str());
  TrainConfig cfg = config(8, 2);
  cfg.trace_words_path = path;
  const TrainResult r = train_text(text, cfg);
  EXPECT_EQ(r.merges, train_text(text, config(1, 0)).merges);  // the sample changes nothing

  const WordCounts counts = count_words(text, cfg);
  const std::string json = read_file(path);
  EXPECT_NE(json.find("\"owners\": 2, \"merge_threads\": 2"), std::string::npos)
      << json.substr(0, 80);
  std::istringstream lines(json);
  std::string line;
  size_t n = 0;
  while (std::getline(lines, line)) {
    unsigned w = 0, owner = 0;
    unsigned long long count = 0;
    char hex[256] = {};
    if (std::sscanf(line.c_str(),
                    "{\"w\": %u, \"count\": %llu, \"owner\": %u, \"hex\": \"%255[0-9a-f]", &w,
                    &count, &owner, hex) != 4) {
      continue;
    }
    ++n;
    ASSERT_LT(w, counts.words.size());
    EXPECT_EQ(unhex(hex), counts.words[w]);
    EXPECT_EQ(count, counts.freq[w]);
    EXPECT_EQ(owner, detail::hashed_owner(w, 2));
  }
  EXPECT_GT(n, 50u);
  std::remove(path.c_str());
}
