// End to end against the Python reference: merges.txt and vocab.json must match byte for byte.

#include <gtest/gtest.h>

#include <vector>

#include "golden_case.hpp"
#include "parabpe/io.hpp"
#include "parabpe/trainer.hpp"

using namespace parabpe;

namespace {

// Every implemented merge engine; add each new one here (S6/S7: parallel).
const std::vector<MergeStrategy> kEngines = {MergeStrategy::NaiveSerial, MergeStrategy::IncrSerial};

}  // namespace

TEST(Golden, MergesAndVocabMatchReference) {
  const auto cases = test::golden_cases();
  ASSERT_FALSE(cases.empty()) << "no golden cases in " << PARABPE_GOLDEN_DIR;
  for (MergeStrategy engine : kEngines) {
    for (const auto& gc : cases) {
      SCOPED_TRACE(gc.name + " / engine " + std::to_string(static_cast<int>(engine)));
      TrainConfig cfg;
      cfg.vocab_size = gc.vocab_size;
      cfg.special_tokens = gc.specials;
      cfg.merge = engine;
      const std::string text = test::read_file(gc.dir / "input.txt");
      const TrainResult r = train_text(text, cfg);
      EXPECT_EQ(format_merges(r.merges, r.vocab), test::read_file(gc.dir / "merges.txt"));
      EXPECT_EQ(format_vocab(r.vocab, cfg.special_tokens.size()),
                test::read_file(gc.dir / "vocab.json"));
    }
  }
}
