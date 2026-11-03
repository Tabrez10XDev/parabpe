#pragma once

// Reads the golden cases written by reference/make_golden.py (see reference/README.md).

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace parabpe::test {

struct GoldenCase {
  std::string name;
  std::filesystem::path dir;
  size_t vocab_size = 0;
  std::vector<std::string> specials;  // in config order
};

inline std::string read_file(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path.string());
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// case.txt: "vocab_size <N>" then one "special <token>" line per special token.
inline GoldenCase read_case(const std::filesystem::path& dir) {
  GoldenCase c;
  c.name = dir.filename().string();
  c.dir = dir;
  std::istringstream lines(read_file(dir / "case.txt"));
  for (std::string line; std::getline(lines, line);) {
    const size_t space = line.find(' ');
    const std::string key = line.substr(0, space);
    const std::string value = space == std::string::npos ? "" : line.substr(space + 1);
    if (key == "vocab_size") c.vocab_size = std::stoul(value);
    if (key == "special") c.specials.push_back(value);
  }
  return c;
}

inline std::vector<GoldenCase> golden_cases() {
  std::vector<GoldenCase> cases;
  for (const auto& entry : std::filesystem::directory_iterator(PARABPE_GOLDEN_DIR)) {
    if (std::filesystem::exists(entry.path() / "case.txt"))
      cases.push_back(read_case(entry.path()));
  }
  std::sort(cases.begin(), cases.end(),
            [](const GoldenCase& a, const GoldenCase& b) { return a.name < b.name; });
  return cases;
}

}  // namespace parabpe::test
