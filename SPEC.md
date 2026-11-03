# ParaBPE Training Spec (frozen)

Every engine follows this spec exactly: serial, OpenMP, and the Python reference
(`reference/bpe_ref.py`). Exactness means byte-identical `merges.txt` across all
strategies and thread counts, and it depends on this document. Changes need agreement
from the whole team, plus regenerated golden files.

## 1. Input normalization (done once, by `scripts/slice_data.py`)
- A corpus is a sequence of documents separated by the special token `<|endoftext|>`.
  For corpora without separators (WikiText, enwik9), the prep script inserts one between articles.
- Invalid UTF-8 is replaced with U+FFFD during prep, so every engine sees valid UTF-8.
  The C++ code validates once, then matches with `PCRE2_NO_UTF_CHECK`.

## 2. Special tokens & chunking
- Split the corpus on special tokens and **drop them before the regex runs**. Special
  tokens get fixed ids after the 256 byte tokens and never take part in merges.
- **The unit of work is a document.** Documents are grouped into ~1 MB work units by a
  deterministic rule that does **not** depend on the thread count. Threads pull work units dynamically.
- The regex never sees text across a document boundary, in any engine. Splitting on
  `\n\n` or other whitespace is not allowed, because `\s+(?!\S)` would split differently.

### Pre-tokenization pattern (GPT-2)
```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```
Character classes (`\s`, `\p{L}`, `\p{N}`) follow **Unicode 16.0**: PCRE2 10.45 in C++ (compiled with
`PCRE2_UTF | PCRE2_UCP`), and Python `regex==2025.9.18`. Other versions are allowed only if they report Unicode 16.0.

## 3. Pair counting & merge application
- **Counting:** count every adjacent pair, including overlapping ones (`aaa` holds two
  `(a,a)` pairs), weighted by the word's frequency.
- **Applying a merge `(x,y) → z` to a word:** greedy, left to right, without overlaps.
  `aaaa → z z`, and `aaa → z a`.
- **Count update for each affected word (diff the whole word):** subtract every old pair of the word × freq,
  rewrite the word, then add every new pair × freq. Never patch neighbours one at a time.

## 4. Tie-break
Best pair = highest count. On a tie, take the lexicographically largest **tuple**
`(bytes(a), bytes(b))`, compared as a tuple and not as joined bytes. The heap
comparator, the parallel max-reduction and `bpe_ref.py` all implement this exact rule.

## 5. Output format
- `merges.txt` and `vocab.json` use GPT-2's `bytes_to_unicode` mapping, so spaces and
  newlines in tokens survive the file format.
- These files can be loaded directly by Hugging Face `tokenizers` (ByteLevel BPE).
- `timings.json` records the phase times, CPU model, physical and logical core counts, SMT state, git SHA and the full config.

## 6. File formats (contracts shared by C++, Python and the bench harness)

### `merges.txt`
```
#version: 0.2
Ġ t
h e
...
```
- The first line is the header. Then one merge per line, in training order: `A B`, where `A` and `B` are
  token bytes mapped through `bytes_to_unicode`, joined by a single space. The file ends with a newline.
- `new_id` of merge `i` (0-based) = `256 + n_special + i`.

### `vocab.json`
- A JSON object `{ "<token string>": id, ... }`. Byte tokens and merged tokens are written through
  `bytes_to_unicode`. Special tokens are written literally.
- Ids: `0..255` = byte `i`; then the special tokens in config order; then the merges in order.

### `words.tsv` (debug dump of the phase 1 output, used to check S1)
- One line per unique word: `<lowercase hex of word bytes>\t<count>`, sorted by the raw bytes
  in ascending order. `WordCounts` is also produced in this order (sorted by bytes), so results are deterministic.

### `timings.json`
```json
{
  "merges": 31743,
  "timings": {"chunk": 0.0, "pretok": 0.0, "build": 0.0, "pairs": 0.0, "merge": 0.0, "total": 0.0},
  "config": {"input": "...", "vocab_size": 32000, "threads": 16, "pretok": "omp",
             "merge": "incr-par", "par_threshold_tokens": 4096, "work_unit_bytes": 1048576},
  "hardware": {"cpu_model": "...", "physical_cores": 16, "logical_cores": 32, "smt": true,
               "os": "..."},
  "git_sha": "..."
}
```
Times are wall-clock seconds measured with `omp_get_wtime()`.
