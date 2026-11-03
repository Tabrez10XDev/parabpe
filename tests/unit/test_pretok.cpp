// Regex/Matcher, the serial pre-tokenizer, words.tsv, and the golden words.tsv files.

#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "golden_case.hpp"
#include "parabpe/corpus.hpp"
#include "parabpe/io.hpp"
#include "parabpe/pretokenizer.hpp"
#include "parabpe/regex.hpp"

using namespace parabpe;

namespace {

std::vector<std::string> split(const std::string& pattern, std::string_view text) {
  const Regex re(pattern);
  Matcher m(re);
  std::vector<std::string> out;
  m.for_each_match(text, [&out](std::string_view piece) { out.emplace_back(piece); });
  return out;
}

WordCounts pretokenize(std::string_view text, const std::vector<std::string>& specials,
                       PretokStrategy strategy = PretokStrategy::Serial) {
  TrainConfig cfg;
  cfg.special_tokens = specials;
  const auto units = make_work_units(text, specials, cfg.work_unit_bytes);
  return make_pretokenizer(strategy)->run(units, cfg);
}

std::string words_tsv(const WordCounts& counts) {
  return format_words(counts);
}

}  // namespace

TEST(Regex, GPT2PatternSplitsLikeReference) {
  // Matches Python: regex.findall(GPT2_PAT, "Hello world's  test 123 ünï!!")
  EXPECT_EQ(split(kGPT2Pattern, "Hello world's  test 123 \xC3\xBCn\xC3\xAF!!"),
            (std::vector<std::string>{"Hello", " world", "'s", " ", " test", " 123",
                                      " \xC3\xBCn\xC3\xAF", "!!"}));
}

TEST(Regex, TrailingWhitespaceLookahead) {
  EXPECT_EQ(split(kGPT2Pattern, "a\n\n  b"), (std::vector<std::string>{"a", "\n\n ", " b"}));
  EXPECT_EQ(split(kGPT2Pattern, "a  "), (std::vector<std::string>{"a", "  "}));
}

TEST(Regex, LongRunsAreSingleMatches) {
  // ~1 MB runs must not hit JIT stack or match limits (a silent stop would drop text).
  const std::string spaces(1 << 20, ' '), letters(1 << 20, 'x'), digits(1 << 20, '7');
  EXPECT_EQ(split(kGPT2Pattern, spaces), std::vector<std::string>{spaces});
  EXPECT_EQ(split(kGPT2Pattern, letters), std::vector<std::string>{letters});
  EXPECT_EQ(split(kGPT2Pattern, " " + digits), std::vector<std::string>{" " + digits});
  // Whitespace before a word: all but the last space, then " b".
  EXPECT_EQ(split(kGPT2Pattern, "a" + spaces + "b"),
            (std::vector<std::string>{"a", spaces.substr(1), " b"}));
}

TEST(Regex, GapThrowsInsteadOfDroppingText) {
  EXPECT_THROW(split("a", "aab"), std::runtime_error);  // 'b' is not covered
  EXPECT_THROW(split("a", "ba"), std::runtime_error);   // match starts after the gap
}

TEST(Regex, EmptyMatchThrows) {
  EXPECT_THROW(split("a*", "b"), std::runtime_error);
}

TEST(Regex, BadPatternThrows) {
  EXPECT_THROW(Regex("(unclosed"), std::runtime_error);
}

TEST(SerialPreTokenizer, CountsAcrossDocuments) {
  const WordCounts c =
      pretokenize("the cat<|endoftext|>the dog<|endoftext|>the", {"<|endoftext|>"});
  EXPECT_EQ(c.words, (std::vector<std::string>{" cat", " dog", "the"}));
  EXPECT_EQ(c.freq, (std::vector<uint64_t>{1, 1, 3}));
}

TEST(SerialPreTokenizer, RegexNeverCrossesDocumentBoundary) {
  // Without the boundary, "a" and "b" would be "a", " b"; with it, " " ends doc 1 as its own word.
  const WordCounts c = pretokenize("a <|endoftext|>b", {"<|endoftext|>"});
  EXPECT_EQ(c.words, (std::vector<std::string>{" ", "a", "b"}));
}

TEST(SerialPreTokenizer, SortsByUnsignedBytes) {
  // "é" (0xC3 0xA9) must sort after "z" (0x7A): bytes compare unsigned, as in Python.
  const WordCounts c = pretokenize("\xC3\xA9 z", {});
  EXPECT_EQ(c.words, (std::vector<std::string>{" z", "\xC3\xA9"}));
}

TEST(WriteWords, Format) {
  WordCounts c;
  c.words = {" a", "\n", "\xC3\xA9"};
  c.freq = {3, 1, 12};
  EXPECT_EQ(words_tsv(c), "2061\t3\n0a\t1\nc3a9\t12\n");
  EXPECT_EQ(words_tsv(WordCounts{}), "");
}

TEST(Golden, WordsTsvMatchesReference) {
  const auto cases = test::golden_cases();
  ASSERT_FALSE(cases.empty()) << "no golden cases in " << PARABPE_GOLDEN_DIR;
  for (const auto& gc : cases) {
    SCOPED_TRACE(gc.name);
    const std::string text = test::read_file(gc.dir / "input.txt");
    EXPECT_EQ(words_tsv(pretokenize(text, gc.specials)), test::read_file(gc.dir / "words.tsv"));
  }
}
