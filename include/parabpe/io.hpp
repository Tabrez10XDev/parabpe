#pragma once

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "parabpe/types.hpp"

namespace parabpe {

// GPT-2 byte -> printable unicode codepoint mapping, UTF-8 encoded (SPEC.md §5).
const std::array<std::string, 256>& bytes_to_unicode();

// Raw token bytes -> their bytes_to_unicode string (the form used in merges.txt and vocab.json).
std::string to_unicode_string(std::string_view bytes);

// Id -> bytes: 256 byte tokens, then the special tokens, then one token per merge (SPEC.md §6).
Vocab build_vocab(const std::vector<Merge>& merges, const std::vector<std::string>& special_tokens);

// File contents in the exact SPEC.md §6 formats (byte rules in reference/README.md).
// words.tsv: `<lowercase hex of word bytes>\t<count>\n`, in the given order (sorted by bytes).
std::string format_words(const WordCounts& counts);
// merges.txt: "#version: 0.2\n" + one "A B\n" line per merge.
std::string format_merges(const std::vector<Merge>& merges, const Vocab& vocab);
// vocab.json: compact JSON like Python's json.dumps(ensure_ascii=False, separators=(",", ":")),
// keys in id order, no trailing newline. Throws if two ids map to the same key.
std::string format_vocab(const Vocab& vocab, size_t n_special);
// timings.json: phase times, config, hardware and git SHA.
std::string format_timings(const TrainResult& result, const std::string& input,
                           const TrainConfig& cfg);

// JSON string literal (with quotes), escaped like Python's json.dumps(ensure_ascii=False).
std::string json_string(std::string_view s);

void write_file(const std::string& path, std::string_view contents);

}  // namespace parabpe
