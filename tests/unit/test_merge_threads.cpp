// B8: --merge-threads. Phase 3 runs on its own team size; phase 1 keeps --threads. The merges
// must not change, and the trace shows which threads did the phase-3 work.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
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

// Every round parallel (threshold 0), so each round records one OWNER_MERGE per owner.
TrainConfig config(MergeStrategy merge, int threads, int merge_threads) {
  TrainConfig cfg;
  cfg.vocab_size = 256 + 1 + 120;
  cfg.merge = merge;
  cfg.pretok = PretokStrategy::OMP;
  cfg.threads = threads;
  cfg.merge_threads = merge_threads;
  cfg.owners_per_thread = 1;
  cfg.par_threshold_tokens = 0;
  cfg.batch_skip_small = false;
  return cfg;
}

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

bool is_phase3(TraceKind k) {
  return k == TraceKind::OwnerMerge || k == TraceKind::Select || k == TraceKind::SerialApply ||
         k == TraceKind::OrderUpdate || k == TraceKind::ZeroSweep;
}

// Highest thread number that recorded a phase-3 event, or -1.
int last_phase3_thread(const Tracer& tracer) {
  int last = -1;
  for (int t = 0; t < tracer.threads(); ++t) {
    for (const detail::TraceEvent& e : tracer.events(t)) {
      if (is_phase3(e.kind)) last = t;
    }
  }
  return last;
}

size_t occurrences(const std::string& s, const std::string& needle) {
  size_t n = 0;
  for (size_t p = s.find(needle); p != std::string::npos; p = s.find(needle, p + 1)) ++n;
  return n;
}

// Which threads pick up owners is up to the OpenMP scheduler (under TSan thread 0 can take them
// all), so the checks are deterministic: incr-batch has one owner per merge thread and records one
// OWNER_MERGE per owner per parallel round, and no phase-3 event runs above the merge team.
void expect_merge_team(const std::string& text, int threads, int merge_threads) {
  for (MergeStrategy m : {MergeStrategy::IncrBatch, MergeStrategy::IncrPar}) {
    const TrainConfig cfg = config(m, threads, merge_threads);
    Tracer tracer(std::max(threads, merge_threads));
    const TrainResult r = traced(text, cfg, tracer);
    EXPECT_EQ(r.merges, train_text(text, config(m, 1, 0)).merges);
    EXPECT_EQ(tracer.dropped(), 0u);
    EXPECT_GT(tracer.count(TraceKind::OwnerMerge), 0u);
    EXPECT_LT(last_phase3_thread(tracer), merge_threads);
    if (m == MergeStrategy::IncrBatch) {
      ASSERT_GT(r.stats.parallel_rounds, 0u);
      EXPECT_EQ(tracer.count(TraceKind::OwnerMerge), r.stats.parallel_rounds * merge_threads);
    }
  }
}

}  // namespace

TEST(MergeThreads, Resolve) {
  TrainConfig cfg;
  cfg.threads = 11;
  EXPECT_EQ(resolve_merge_threads(cfg, 5), 11);  // 0 = same as threads
  cfg.merge_threads = 3;
  EXPECT_EQ(resolve_merge_threads(cfg, 5), 3);
  cfg.merge_threads = 16;
  EXPECT_EQ(resolve_merge_threads(cfg, 5), 16);  // an explicit count may exceed threads
  cfg.merge_threads = kMergeThreadsAuto;
  EXPECT_EQ(resolve_merge_threads(cfg, 5), 5);
  EXPECT_EQ(resolve_merge_threads(cfg, -1), 11);  // P-cores unknown: all threads
  cfg.threads = 3;
  EXPECT_EQ(resolve_merge_threads(cfg, 5), 3);  // auto never exceeds threads
  cfg.merge_threads = -7;
  EXPECT_THROW(resolve_merge_threads(cfg, 5), std::invalid_argument);
}

TEST(MergeThreads, AutoMatchesThisMachine) {
  TrainConfig cfg;
  cfg.threads = 64;
  cfg.merge_threads = kMergeThreadsAuto;
  const int p = performance_cores();
  EXPECT_EQ(resolve_merge_threads(cfg), p >= 1 ? p : 64);
}

TEST(MergeThreads, WiderTeamForPhase3) {
  expect_merge_team(random_text(11, 8000), 2, 4);
}

TEST(MergeThreads, NarrowerTeamForPhase3) {
  expect_merge_team(random_text(12, 8000), 8, 2);
}

// --trace sizes its per-thread buffers for the wider of the two teams.
TEST(MergeThreads, TraceFileCoversTheWiderTeam) {
  const std::string path = ::testing::TempDir() + "parabpe_merge_threads.json";
  std::remove(path.c_str());
  TrainConfig cfg = config(MergeStrategy::IncrBatch, 2, 4);
  cfg.trace_path = path;
  train_text(random_text(13, 8000), cfg);
  std::ifstream in(path, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string json = ss.str();
  EXPECT_NE(json.find("\"dropped\": 0"), std::string::npos);
  EXPECT_EQ(occurrences(json, "\"name\": \"thread_name\""), 4u);
  std::remove(path.c_str());
}
