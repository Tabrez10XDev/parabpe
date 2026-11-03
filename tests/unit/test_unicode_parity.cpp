// PCRE2 vs Python `regex` on every code point, for the three classes the GPT-2 pattern uses.
// tests/golden/unicode_classes.txt is written by reference/make_golden.py (SPEC.md §2:
// Unicode 16.0).

#include <gtest/gtest.h>
#include <pcre2.h>

#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "golden_case.hpp"
#include "parabpe/regex.hpp"

namespace {

constexpr uint32_t kMaxCodePoint = 0x10FFFF;

std::string utf8(uint32_t cp) {
  std::string s;
  if (cp < 0x80) {
    s += char(cp);
  } else if (cp < 0x800) {
    s += char(0xC0 | (cp >> 6));
    s += char(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    s += char(0xE0 | (cp >> 12));
    s += char(0x80 | ((cp >> 6) & 0x3F));
    s += char(0x80 | (cp & 0x3F));
  } else {
    s += char(0xF0 | (cp >> 18));
    s += char(0x80 | ((cp >> 12) & 0x3F));
    s += char(0x80 | ((cp >> 6) & 0x3F));
    s += char(0x80 | (cp & 0x3F));
  }
  return s;
}

// class name -> membership per code point, from the Python-generated ranges.
std::map<std::string, std::vector<bool>> load_python_classes() {
  std::map<std::string, std::vector<bool>> classes;
  std::istringstream in(
      parabpe::test::read_file(std::string(PARABPE_GOLDEN_DIR) + "/unicode_classes.txt"));
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::string name;
    uint32_t first = 0, last = 0;
    fields >> name >> std::hex >> first >> last;
    auto& bits = classes[name];
    bits.resize(kMaxCodePoint + 1);
    for (uint32_t cp = first; cp <= last; ++cp) bits[cp] = true;
  }
  return classes;
}

}  // namespace

TEST(UnicodeParity, WhitespaceRewrite) {
  EXPECT_EQ(parabpe::unicode_whitespace_pattern(R"([^\s\p{L}]|\s+(?!\S)|\\s)"),
            R"([^\p{White_Space}\p{L}]|\p{White_Space}+(?!\P{White_Space})|\\s)");
}

TEST(UnicodeParity, MongolianVowelSeparatorIsNotWhitespace) {
  // Raw PCRE2 \s says yes (legacy list); Unicode 16 and Python's regex say no.
  const parabpe::Regex space(R"(\s)");
  parabpe::Matcher m(space);
  EXPECT_THROW(m.match_at("\xE1\xA0\x8E", 0), std::runtime_error);  // U+180E: no match
  EXPECT_EQ(m.match_at("\xC2\x85", 0), 2u);                         // U+0085 NEL: whitespace
}

TEST(UnicodeParity, Pcre2MatchesPythonRegexOnEveryCodePoint) {
  const auto classes = load_python_classes();
  ASSERT_EQ(classes.size(), 3u);
  const std::map<std::string, std::string> patterns = {
      {"s", R"(\s)"}, {"L", R"(\p{L})"}, {"N", R"(\p{N})"}};

  for (const auto& [name, pattern] : patterns) {
    SCOPED_TRACE("class " + name);
    ASSERT_TRUE(classes.count(name));
    const auto& expected = classes.at(name);

    // Through parabpe::Regex, so the \s -> \p{White_Space} rewrite is what gets checked.
    const parabpe::Regex regex(pattern);
    const pcre2_code* re = regex.code();
    pcre2_match_data* md = pcre2_match_data_create_from_pattern(re, nullptr);

    int mismatches = 0;
    for (uint32_t cp = 0; cp <= kMaxCodePoint; ++cp) {
      if (cp >= 0xD800 && cp <= 0xDFFF) continue;  // surrogates are not valid UTF-8
      const std::string s = utf8(cp);
      const int rc = pcre2_match(re, reinterpret_cast<PCRE2_SPTR>(s.data()), s.size(), 0,
                                 PCRE2_ANCHORED | PCRE2_NO_UTF_CHECK, md, nullptr);
      ASSERT_TRUE(rc > 0 || rc == PCRE2_ERROR_NOMATCH) << "PCRE2 error " << rc;
      const bool pcre2_says = rc > 0;
      if (pcre2_says != expected[cp] && ++mismatches <= 10) {
        ADD_FAILURE() << "U+" << std::hex << std::uppercase << cp << ": PCRE2=" << pcre2_says
                      << " Python=" << expected[cp];
      }
    }
    EXPECT_EQ(mismatches, 0);
    pcre2_match_data_free(md);
  }
}
