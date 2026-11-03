# Python reference (`bpe_ref.py`)

A short, slow, spec-exact oracle for `SPEC.md`. Every C++ engine must match its outputs byte for byte.

## Versions (these matter for exactness)
| Component | Version | Why it matters |
|---|---|---|
| Python | 3.13.3 | |
| `regex` | **2025.9.18** (pinned in `requirements.txt`) | Ships its own Unicode tables. This is the last release on **Unicode 16.0**; 2025.11.3+ is Unicode 17. |
| PCRE2 (C++ side) | **10.45** | Unicode 16.0.0 (`PCRE2_CONFIG_UNICODE_VERSION`) |
| `tokenizers` | 0.23.3 | Only for the HF interop test |

SPEC.md §2 pins Unicode 16.0. One PCRE2 quirk: with UCP, its `\s` uses a legacy list that includes U+180E
(not whitespace since Unicode 6.3). The C++ `Regex` therefore compiles `\s` as `\p{White_Space}`, which matches
`regex` exactly; `test_unicode_parity.cpp` checks all three classes on every code point, and `whitespace_zoo` holds U+180E. Two tests guard it: `test_unicode_version_is_16` (Python) and
`PCRE2.UnicodeVersionIsPinned` (C++). The golden case `unicode16` holds letters that are new in Unicode 16
(Garay, Ol Onal) and one that is new in 17 (Sidetic), so a version drift on either side changes `words.tsv`.

## Usage
```bash
.venv/bin/python reference/bpe_ref.py --input F --vocab-size 1000 --out runs/ref     # words.tsv, merges.txt, vocab.json
.venv/bin/python reference/bpe_ref.py --input F --words-only --out runs/ref          # phase 1 only (fast)
  [--special S]...   # repeatable; default <|endoftext|> applies only when no --special is given
  [--no-special]     # no special tokens at all

.venv/bin/python reference/make_golden.py            # regenerate tests/golden/
.venv/bin/python reference/make_golden.py --check    # exit 1 if tests/golden/ drifted
.venv/bin/python -m unittest discover reference      # tests, incl. HF token-id check
```

## Golden cases (`tests/golden/<case>/`)
Each holds `input.txt`, `case.txt` (line-based config: `vocab_size N`, then one `special <token>` line per special
token, in order), and the expected `words.tsv`, `merges.txt` and `vocab.json`.

| Case | Covers |
|---|---|
| `unicode` | Accents, CJK, Cyrillic, Arabic, Devanagari (combining marks), Greek, Hangul |
| `emoji` | Skin-tone modifiers, ZWJ sequences, flags, emoji glued to text |
| `runs` | `aaaa`, odd/even runs, `abab…`, long punctuation and digit runs |
| `ties` | Many equal-count pairs, so the tie-break decides |
| `multi_occurrence` | One pair several times in a word (`abcabc`, `banana`) |
| `doc_edges` | Whitespace at document edges, empty docs, whitespace-only docs, trailing separator |
| `no_separator` | A single document |
| `multi_special` | Specials `<|endoftext|>` + `<|end` (a prefix of it): leftmost match, longest first |
| `early_stop` | `vocab_size` larger than the possible number of merges |
| `whitespace_zoo` | `\r\n`, `\v`, `\f`, U+001C–U+001F, U+0085, U+00A0, U+2028/9, U+3000, U+200B, BOM, `’s`, combining accents, `½²`, fullwidth digits |
| `unicode16` | Unicode 16 letters and digits, and one Unicode 17 letter |

`tests/golden/unicode_classes.txt` lists the code-point ranges of `\s`, `\p{L}` and `\p{N}` according to `regex`, for
the C++ parity test (S1).

## Byte-exact output rules (the C++ writers must reproduce these)
- `words.tsv`: `<lowercase hex>\t<count>\n` per word, sorted by raw bytes (unsigned).
- `merges.txt`: `#version: 0.2\n`, then `A B\n` per merge in training order (tokens through `bytes_to_unicode`).
  An empty corpus gives the header only.
- `vocab.json`: Python `json.dumps(table, ensure_ascii=False, separators=(",", ":"))`, keys in id order, UTF-8,
  **no trailing newline**. Escapes: `"` → `\"`, `\` → `\\`, `\n` `\r` `\t` `\b` `\f` → short escapes, other
  U+0000–U+001F → `\u00xx` (lowercase hex). Everything else (including U+007F, U+2028, U+2029) is written raw.
  In practice only special tokens can contain these, because `bytes_to_unicode` maps control bytes to printable characters.
- Input is read in binary and decoded strictly; no newline translation, the BOM is kept as U+FEFF.

## Notes
- Training recounts every pair on every merge (naive). That is fine for golden cases; use `--words-only` on big inputs.
- `train` asserts that no merge produces bytes that already exist in the vocab. About 27k random trials never hit it,
  and greedy left-to-right merging seems to rule it out; if it ever fires, SPEC.md needs a rule.
- `encode` is a plain rank-based GPT-2 encoder, used to check our files against HF `tokenizers` by token ids
  (a decode round trip alone always passes for byte-level BPE).
