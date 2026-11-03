// Toolchain smoke tests: OpenMP, PCRE2 Unicode version, mmap corpus, factories.

#include <gtest/gtest.h>
#include <omp.h>
#include <pcre2.h>

#include <ankerl/unordered_dense.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "parabpe/corpus.hpp"
#include "parabpe/merge_engine.hpp"
#include "parabpe/pretokenizer.hpp"
#include "parabpe/types.hpp"

using namespace parabpe;

TEST(OpenMP, ParallelReductionMatchesSerial) {
  constexpr int64_t n = 1'000'000;
  int64_t sum = 0;
#pragma omp parallel for reduction(+ : sum)
  for (int64_t i = 1; i <= n; ++i) sum += i;
  EXPECT_EQ(sum, n * (n + 1) / 2);
  EXPECT_GE(omp_get_max_threads(), 1);
}

TEST(OpenMP, ThreadLocalMapsMergeToSameCounts) {
  const std::vector<int> data = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
  ankerl::unordered_dense::map<int, int> merged;
#pragma omp parallel
  {
    ankerl::unordered_dense::map<int, int> local;
#pragma omp for nowait
    for (size_t i = 0; i < data.size(); ++i) ++local[data[i]];
#pragma omp critical
    for (auto& [k, v] : local) merged[k] += v;
  }
  for (int k = 1; k <= 4; ++k) EXPECT_EQ(merged[k], k);
}

TEST(PCRE2, UnicodeVersionIsPinned) {
  // SPEC.md §2: character classes follow Unicode 16.0 (Python side: regex==2025.9.18).
  char version[32] = {};
  ASSERT_GT(pcre2_config(PCRE2_CONFIG_UNICODE_VERSION, version), 0);
  EXPECT_STREQ(version, "16.0.0");
}

TEST(Corpus, MmapReadsFile) {
  const auto path = std::filesystem::temp_directory_path() / "parabpe_corpus_test.txt";
  {
    std::ofstream f(path, std::ios::binary);
    f << "doc one<|endoftext|>doc two";
  }
  Corpus c = Corpus::open(path.string());
  EXPECT_EQ(c.text(), "doc one<|endoftext|>doc two");
  Corpus moved = std::move(c);
  EXPECT_EQ(moved.text().size(), 27u);
  EXPECT_TRUE(c.text().empty());
  std::filesystem::remove(path);
}

TEST(Factories, AllStrategiesConstruct) {
  for (auto s : {PretokStrategy::Serial, PretokStrategy::OMP}) EXPECT_NE(make_pretokenizer(s), nullptr);
  for (auto s : {MergeStrategy::NaiveSerial, MergeStrategy::IncrSerial, MergeStrategy::NaivePar,
                 MergeStrategy::IncrPar})
    EXPECT_NE(make_merge_engine(s), nullptr);
}

TEST(Types, PairKeyRoundTrip) {
  constexpr PairKey k = make_pair_key(70000, 42);
  static_assert(pair_first(k) == 70000 && pair_second(k) == 42);
  EXPECT_LT(make_pair_key(1, 0xffffffffu), make_pair_key(2, 0));
}
