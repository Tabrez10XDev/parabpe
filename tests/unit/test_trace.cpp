// B6: --trace (src/trace.hpp). Tracing must not change the merges, must record the events the
// engine's own statistics predict, and must be off unless asked for.

#include <gtest/gtest.h>

#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "parabpe/trainer.hpp"
#include "trace.hpp"

using namespace parabpe;
using detail::TraceKind;
using detail::Tracer;

namespace {

std::string random_text(unsigned seed, int n) {
  std::mt19937 rng(seed);
  const std::string alphabet = "abcdefgh  \n";
  std::string text;
  for (int i = 0; i < n; ++i) {
    text += alphabet[rng() % alphabet.size()];
    if (i % 400 == 399) text += "<|endoftext|>";
  }
  return text;
}

TrainConfig batch_config(int threads, size_t threshold) {
  TrainConfig cfg;
  cfg.vocab_size = 256 + 1 + 120;
  cfg.merge = MergeStrategy::IncrBatch;
  cfg.pretok = PretokStrategy::OMP;
  cfg.threads = threads;
  cfg.owners_per_thread = 1;
  cfg.par_threshold_tokens = threshold;
  cfg.batch_skip_small = false;
  return cfg;
}

// Runs one training with `tracer` installed directly, so the test can read its events.
TrainResult traced(const std::string& text, const TrainConfig& cfg, Tracer& tracer) {
  detail::g_tracer = &tracer;
  TrainResult r;
  try {
    r = train_text(text, cfg);
  } catch (...) {
    detail::g_tracer = nullptr;
    throw;
  }
  detail::g_tracer = nullptr;
  return r;
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

size_t occurrences(const std::string& s, const std::string& needle) {
  size_t n = 0;
  for (size_t p = s.find(needle); p != std::string::npos; p = s.find(needle, p + 1)) ++n;
  return n;
}

}  // namespace

TEST(Trace, OffByDefault) {
  EXPECT_EQ(detail::g_tracer, nullptr);
  const TrainResult r = train_text(random_text(1, 3000), batch_config(2, 0));
  EXPECT_FALSE(r.merges.empty());
  EXPECT_EQ(detail::g_tracer, nullptr);
}

// All rounds parallel: one OWNER_MERGE per owner per round, no SERIAL_APPLY.
TEST(Trace, ParallelRoundsMatchStats) {
  const std::string text = random_text(2, 6000);
  const int threads = 3;
  const TrainConfig cfg = batch_config(threads, 0);
  Tracer tracer(threads);
  const TrainResult r = traced(text, cfg, tracer);

  EXPECT_EQ(r.merges, train_text(text, cfg).merges);
  ASSERT_GT(r.stats.rounds, 0u);
  EXPECT_EQ(r.stats.parallel_rounds, r.stats.rounds);
  const bool exhausted = r.merges.size() < cfg.vocab_size - 257;
  EXPECT_EQ(tracer.count(TraceKind::Select), r.stats.rounds + (exhausted ? 1 : 0));
  EXPECT_EQ(tracer.count(TraceKind::OrderUpdate), r.stats.rounds);
  EXPECT_EQ(tracer.count(TraceKind::OwnerMerge), r.stats.parallel_rounds * threads);
  EXPECT_EQ(tracer.count(TraceKind::SerialApply), 0u);
  EXPECT_GE(tracer.count(TraceKind::ShardReduce), r.stats.parallel_rounds);
  EXPECT_GE(tracer.count(TraceKind::PretokUnit), 1u);
  EXPECT_GE(tracer.count(TraceKind::Chunk), 1u);
  EXPECT_EQ(tracer.dropped(), 0u);

  uint64_t merged = 0;
  for (int t = 0; t < tracer.threads(); ++t) {
    for (const detail::TraceEvent& e : tracer.events(t)) {
      EXPECT_LE(e.t0, e.t1);
      if (e.kind == TraceKind::Select) merged += e.arg;
      if (e.kind == TraceKind::Select || e.kind == TraceKind::OrderUpdate) EXPECT_EQ(t, 0);
    }
  }
  EXPECT_EQ(merged, r.merges.size());  // SELECT's arg is the round's merge count
}

// No round reaches the threshold: every round is a SERIAL_APPLY on the master.
TEST(Trace, SerialRoundsMatchStats) {
  const std::string text = random_text(3, 6000);
  const TrainConfig cfg = batch_config(2, size_t{1} << 40);
  Tracer tracer(2);
  const TrainResult r = traced(text, cfg, tracer);
  EXPECT_EQ(r.merges, train_text(text, cfg).merges);
  EXPECT_EQ(r.stats.parallel_rounds, 0u);
  EXPECT_EQ(tracer.count(TraceKind::SerialApply), r.stats.rounds);
  EXPECT_EQ(tracer.count(TraceKind::OwnerMerge), 0u);
  for (const detail::TraceEvent& e : tracer.events(1)) {
    EXPECT_TRUE(e.kind == TraceKind::Chunk || e.kind == TraceKind::PretokUnit ||
                e.kind == TraceKind::ShardReduce)
        << detail::trace_name(e.kind) << " outside phase 1 on a worker thread";
  }
}

// The vocabulary is never reached: the last selection finds no pair and is recorded too.
TEST(Trace, ExhaustedRunRecordsTheEmptySelection) {
  TrainConfig cfg = batch_config(2, 0);
  cfg.vocab_size = 256 + 1 + 1000;
  Tracer tracer(2);
  const TrainResult r = traced(random_text(6, 800), cfg, tracer);
  ASSERT_LT(r.merges.size(), cfg.vocab_size - 257);
  EXPECT_EQ(tracer.count(TraceKind::Select), r.stats.rounds + 1);
  EXPECT_EQ(tracer.count(TraceKind::OrderUpdate), r.stats.rounds);
}

TEST(Trace, IncrParRecordsOwnerMerges) {
  const std::string text = random_text(4, 6000);
  TrainConfig cfg = batch_config(2, 0);
  cfg.merge = MergeStrategy::IncrPar;
  Tracer tracer(2);
  const TrainResult r = traced(text, cfg, tracer);
  EXPECT_EQ(r.merges, train_text(text, cfg).merges);
  EXPECT_GT(tracer.count(TraceKind::OwnerMerge), 0u);
  EXPECT_EQ(tracer.count(TraceKind::Select), 0u);
}

TEST(Trace, WritesChromeJson) {
  const std::string path = ::testing::TempDir() + "parabpe_trace_test.json";
  std::remove(path.c_str());
  const std::string text = random_text(5, 6000);
  TrainConfig cfg = batch_config(2, 0);
  cfg.trace_path = path;
  const TrainResult r = train_text(text, cfg);
  EXPECT_EQ(detail::g_tracer, nullptr);

  const std::string json = read_file(path);
  ASSERT_FALSE(json.empty());
  EXPECT_EQ(json.front(), '{');
  EXPECT_EQ(json.substr(json.size() - 4), "\n]}\n");
  EXPECT_NE(json.find("\"traceEvents\""), std::string::npos);
  EXPECT_NE(json.find("\"dropped\": 0"), std::string::npos);
  EXPECT_EQ(occurrences(json, "\"name\": \"SELECT\""), r.stats.rounds);
  EXPECT_EQ(occurrences(json, "\"name\": \"thread_name\""), 2u);
  std::remove(path.c_str());
}

TEST(Trace, UninstalledWhenTrainingThrows) {
  const std::string path = ::testing::TempDir() + "parabpe_trace_throw.json";
  TrainConfig cfg = batch_config(2, 0);
  cfg.trace_path = path;
  cfg.special_tokens = {""};  // rejected by the chunker
  EXPECT_THROW(train_text("abc", cfg), std::invalid_argument);
  EXPECT_EQ(detail::g_tracer, nullptr);
}
