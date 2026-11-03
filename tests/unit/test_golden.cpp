// End to end against the Python reference: merges.txt and vocab.json must match byte for byte.

#include <gtest/gtest.h>

#include <vector>

#include "golden_case.hpp"
#include "parabpe/io.hpp"
#include "parabpe/trainer.hpp"

using namespace parabpe;

namespace {

// Every merge engine. Parallel engines run at every thread count of the equivalence check
// (CLAUDE.md), serial ones once.
const std::vector<MergeStrategy> kEngines = {MergeStrategy::NaiveSerial, MergeStrategy::IncrSerial,
                                             MergeStrategy::NaivePar, MergeStrategy::IncrPar,
                                             MergeStrategy::IncrBatch};
const std::vector<int> kParThreads = {1, 2, 3, 4, 8, 16};

bool is_parallel(MergeStrategy m) {
  return m == MergeStrategy::NaivePar || m == MergeStrategy::IncrPar ||
         m == MergeStrategy::IncrBatch;
}

}  // namespace

TEST(Golden, MergesAndVocabMatchReference) {
  const auto cases = test::golden_cases();
  ASSERT_FALSE(cases.empty()) << "no golden cases in " << PARABPE_GOLDEN_DIR;
  for (MergeStrategy engine : kEngines) {
    for (int threads : is_parallel(engine) ? kParThreads : std::vector<int>{1}) {
      for (const auto& gc : cases) {
        SCOPED_TRACE(gc.name + " / engine " + std::to_string(static_cast<int>(engine)) +
                     " / threads " + std::to_string(threads));
        TrainConfig cfg;
        cfg.vocab_size = gc.vocab_size;
        cfg.special_tokens = gc.specials;
        cfg.merge = engine;
        cfg.threads = threads;
        cfg.par_threshold_tokens = 0;  // IncrPar: every merge step takes the parallel path
        cfg.verify_batches = true;     // IncrBatch: every member against the sequential oracle
        const std::string text = test::read_file(gc.dir / "input.txt");
        const TrainResult r = train_text(text, cfg);
        EXPECT_EQ(format_merges(r.merges, r.vocab), test::read_file(gc.dir / "merges.txt"));
        EXPECT_EQ(format_vocab(r.vocab, cfg.special_tokens.size()),
                  test::read_file(gc.dir / "vocab.json"));
      }
    }
  }
}
