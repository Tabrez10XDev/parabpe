// make_work_units (SPEC.md §2) and the UTF-8 validator.

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "parabpe/corpus.hpp"

using namespace parabpe;

namespace {

const std::vector<std::string> kEot = {"<|endoftext|>"};

std::vector<std::string> all_docs(const std::vector<WorkUnit>& units) {
  std::vector<std::string> out;
  for (const auto& u : units) out.insert(out.end(), u.docs.begin(), u.docs.end());
  return out;
}

std::vector<size_t> unit_sizes(const std::vector<WorkUnit>& units) {
  std::vector<size_t> out;
  for (const auto& u : units) out.push_back(u.docs.size());
  return out;
}

}  // namespace

TEST(Chunker, NoSeparatorIsOneDocument) {
  const auto units = make_work_units("just one document\n", kEot, 1 << 20);
  EXPECT_EQ(all_docs(units), std::vector<std::string>{"just one document\n"});
}

TEST(Chunker, EmptyTextGivesNoUnits) {
  EXPECT_TRUE(make_work_units("", kEot, 1 << 20).empty());
  EXPECT_TRUE(make_work_units("<|endoftext|><|endoftext|>", kEot, 1 << 20).empty());
}

TEST(Chunker, LeadingTrailingAndConsecutiveSeparators) {
  const std::string text =
      "<|endoftext|>  a<|endoftext|><|endoftext|>b <|endoftext|> <|endoftext|>";
  // Empty documents are skipped; whitespace-only documents are kept (same as bpe_ref.split_docs).
  EXPECT_EQ(all_docs(make_work_units(text, kEot, 1 << 20)),
            (std::vector<std::string>{"  a", "b ", " "}));
}

TEST(Chunker, NoSpecialTokensMeansOneDocument) {
  EXPECT_EQ(all_docs(make_work_units("a<|endoftext|>b", {}, 1 << 20)),
            std::vector<std::string>{"a<|endoftext|>b"});
}

TEST(Chunker, LongestSpecialWinsAtSamePosition) {
  const std::string text = "x<|endoftext|>y<|end<|endoftext|>z";
  const std::vector<std::string> expected = {"x", "y", "z"};
  // Config order must not matter.
  EXPECT_EQ(all_docs(make_work_units(text, {"<|end", "<|endoftext|>"}, 1 << 20)), expected);
  EXPECT_EQ(all_docs(make_work_units(text, {"<|endoftext|>", "<|end"}, 1 << 20)), expected);
}

TEST(Chunker, LeftmostSpecialWins) {
  // "ab" starts before "bc"; once "ab" is consumed, "bc" no longer matches.
  EXPECT_EQ(all_docs(make_work_units("1abc2bc3", {"bc", "ab"}, 1 << 20)),
            (std::vector<std::string>{"1", "c2", "3"}));
}

TEST(Chunker, PacksDocumentsUntilTarget) {
  // Docs of 3 bytes, target 7: a unit closes once it reaches >= 7 bytes (3 docs = 9 bytes).
  const std::string text = "aaa|bbb|ccc|ddd|eee";
  const auto units = make_work_units(text, {"|"}, 7);
  EXPECT_EQ(unit_sizes(units), (std::vector<size_t>{3, 2}));
  EXPECT_EQ(all_docs(units), (std::vector<std::string>{"aaa", "bbb", "ccc", "ddd", "eee"}));
}

TEST(Chunker, LargeDocumentGetsItsOwnUnit) {
  const std::string big(100, 'x');
  const std::string text = "a|" + big + "|b|c";  // units hold views, so the text must outlive them
  const auto units = make_work_units(text, {"|"}, 10);
  ASSERT_EQ(unit_sizes(units), (std::vector<size_t>{1, 1, 2}));
  EXPECT_EQ(units[1].docs[0], big);
}

TEST(Chunker, ZeroTargetMeansOneDocumentPerUnit) {
  EXPECT_EQ(unit_sizes(make_work_units("a|b|c", {"|"}, 0)), (std::vector<size_t>{1, 1, 1}));
}

TEST(Chunker, DocumentsAreViewsIntoTheText) {
  const std::string text = "hello<|endoftext|>world";
  const auto units = make_work_units(text, kEot, 1 << 20);
  ASSERT_EQ(units.size(), 1u);
  EXPECT_EQ(units[0].docs[1].data(), text.data() + 18);
}

TEST(Chunker, RejectsBadSpecialTokens) {
  EXPECT_THROW(make_work_units("a", {""}, 1 << 20), std::invalid_argument);
  EXPECT_THROW(make_work_units("a", {"x", "x"}, 1 << 20), std::invalid_argument);
}

TEST(Chunker, RejectsInvalidUtf8) {
  EXPECT_THROW(make_work_units("ok<|endoftext|>bad\xff", kEot, 1 << 20), std::runtime_error);
}

TEST(Utf8, AcceptsValidSequences) {
  for (const std::string s : {"", "plain ascii text, longer than eight bytes",
                              "\xC2\x80",          // U+0080
                              "\xDF\xBF",          // U+07FF
                              "\xE0\xA0\x80",      // U+0800
                              "\xED\x9F\xBF",      // U+D7FF (last before surrogates)
                              "\xEE\x80\x80",      // U+E000
                              "\xEF\xBB\xBF",      // BOM
                              "\xEF\xBF\xBF",      // U+FFFF
                              "\xF0\x90\x80\x80",  // U+10000
                              "\xF0\x9F\x91\x8B\xF0\x9F\x8F\xBD",  // waving hand + skin tone
                              "\xF4\x8F\xBF\xBF"}) {               // U+10FFFF
    EXPECT_NO_THROW(validate_utf8(s)) << testing::PrintToString(s);
  }
}

TEST(Utf8, RejectsInvalidSequences) {
  for (const std::string s : {"\x80",              // stray continuation
                              "\xC0\x80",          // overlong NUL
                              "\xC1\xBF",          // overlong
                              "\xE0\x80\x80",      // overlong 3-byte
                              "\xE0\x9F\xBF",      // overlong 3-byte
                              "\xED\xA0\x80",      // surrogate U+D800
                              "\xED\xBF\xBF",      // surrogate U+DFFF
                              "\xF0\x80\x80\x80",  // overlong 4-byte
                              "\xF0\x8F\xBF\xBF",  // overlong 4-byte
                              "\xF4\x90\x80\x80",  // U+110000
                              "\xF5\x80\x80\x80",  // invalid lead
                              "\xFF",              // invalid lead
                              "\xC3",              // truncated
                              "\xE2\x82",          // truncated
                              "\xE2\x28\xA1",      // bad continuation
                              "\xF0\x9F\x91",      // truncated 4-byte
                              "abcdefgh\xC3"}) {   // truncated after the ASCII fast path
    EXPECT_THROW(validate_utf8(s), std::runtime_error) << testing::PrintToString(s);
  }
}

namespace {

constexpr int kThreads[] = {1, 2, 3, 4, 8, 16};

// Random text of `bytes` bytes (>= the 1 MB parallel cutoff) with dense special tokens, so many
// occurrences straddle the per-thread range borders. "aa" overlaps itself; "<|end" is a prefix of
// "<|endoftext|>".
std::string random_text(size_t bytes, uint32_t seed) {
  std::mt19937 rng(seed);
  // clang-format off
  const std::vector<std::string> pieces = {"a", "aa", "b", " ", "\n", "<|end", "<|endoftext|>",
                                           "\xC3\xA9", "\xE4\xB8\xAD", "\xF0\x9F\x98\x80", "xyz"};
  // clang-format on
  std::string s;
  while (s.size() < bytes) s += pieces[rng() % pieces.size()];
  return s;
}

std::string error_of(std::string_view text, int threads) {
  try {
    validate_utf8(text, threads);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "valid";
}

}  // namespace

TEST(ParallelChunker, UnitsIdenticalToSerial) {
  const std::vector<std::vector<std::string>> special_sets = {
      {"<|endoftext|>"}, {"<|endoftext|>", "<|end"}, {"aa", "<|end"}, {"aa", "a"}, {}};
  const std::string text = random_text(3 << 20, 1);
  for (const auto& specials : special_sets) {
    for (size_t target : {size_t{1000}, size_t{1} << 20}) {
      const auto serial = make_work_units(text, specials, target, 1);
      for (int t : kThreads) {
        SCOPED_TRACE("specials " + std::to_string(specials.size()) + " target " +
                     std::to_string(target) + " threads " + std::to_string(t));
        const auto par = make_work_units(text, specials, target, t);
        ASSERT_EQ(unit_sizes(par), unit_sizes(serial));
        ASSERT_EQ(all_docs(par), all_docs(serial));
      }
    }
  }
}

TEST(ParallelChunker, Utf8ErrorOffsetDoesNotDependOnThreads) {
  const std::string valid = random_text(2 << 20, 2);
  EXPECT_EQ(error_of(valid, 8), "valid");
  // Corrupt bytes at, just before and just after every range border of every thread count, plus
  // a run of continuation bytes longer than any sequence (skipped by the range alignment).
  std::vector<size_t> positions = {0, 1, valid.size() - 1};
  for (int t : kThreads) {
    for (int r = 1; r < t; ++r) {
      const size_t b = valid.size() * r / t;
      for (size_t d : {b - 2, b - 1, b, b + 1}) positions.push_back(d);
    }
  }
  for (size_t p : positions) {
    for (const char* bad : {"\xFF", "\x80", "\xC3", "\x80\x80\x80\x80\x80\x80"}) {
      std::string text = valid;
      text.replace(p, std::min(std::strlen(bad), text.size() - p), bad);
      const std::string expected = error_of(text, 1);
      for (int t : kThreads) {
        ASSERT_EQ(error_of(text, t), expected) << "pos " << p << " threads " << t;
      }
    }
  }
}
