# Handoff: read this first

## Current state (2026-11-03)
**Done:** N1 (repo skeleton), N2 (draft spec and interfaces), **N4 (Python reference + golden files)**, **N3 (data)**, **S1 (chunker + serial pre-tokenizer)** and **S2 (WordTable + NaiveSerial +
writers + `train()`)** **S3 (IncrSerial + equivalence harness)** and **S4 (OMP pre-tokenizer + TSan CI job)**. Training works end to end
with `--pretok serial|omp --merge naive-serial|incr-serial`; the parallel merge engines (S5–S7) are still stubs.

**N4 + Unicode pin (2026-10-31).** The plan for N4 → N3 → S1 was checked by an independent reviewer; its changes are
folded into the S1/S2 cards in `docs/TASKS.md`.
- **SPEC.md §2 now pins Unicode 16.0** (the user approved this). Python `regex==2025.9.18` is the last Unicode-16
  release (2025.11.3+ is Unicode 17), and PCRE2 must be 10.45. `CMakeLists.txt` stops on PCRE2 < 10.45, and the
  gtest `PCRE2.UnicodeVersionIsPinned` checks `16.0.0` at runtime. CI now builds PCRE2 10.45 from source with
  `scripts/build_pcre2.sh` (cached), because Ubuntu's 10.42 is older. That script was run locally and the tests pass
  against it; the CI run itself is unverified (nothing pushed).
  **Watch out:** a `brew upgrade pcre2` to a Unicode-17 release will fail that test. Pin or downgrade if it happens.
- `reference/bpe_ref.py`, `make_golden.py`, `test_bpe_ref.py` and `README.md` are written. There are **11 golden cases**: the 7 from
  the card plus `multi_special`, `early_stop`, `whitespace_zoo` and `unicode16`. There is also `tests/golden/unicode_classes.txt`
  for the S1 parity test. Each case has a line-based `case.txt` (vocab size + specials).
- `.venv/bin/python -m unittest discover reference`: **17/17 pass**. They include an HF `tokenizers` **token-id**
  comparison against `bpe_ref.encode` on held-out text. A mutation check confirmed it fails when merges are dropped or
  reordered. `make_golden.py --check` reports no drift.
- The exact byte format of `vocab.json` (which S2 must match) is in `reference/README.md`.

Verified on this machine (2026-11-03):
- `cmake -B build && cmake --build build -j && ctest --test-dir build`: **53/53 tests** (52 gtest + the `slow`
  equivalence matrix, now 24/24 identical), zero compiler warnings. `ctest -LE slow` skips the matrix. The Debug build (`build-debug`) passes too.
- `.venv/bin/python -m unittest discover reference`: 17/17. `make_golden.py --check`: no drift.
- `./build/parabpe train --input data/tinystories.10M.txt --vocab-size 1000 --out runs/x` works end to end (IncrSerial).

Git: the user made the first commit (`Initial setup`, 2026-11-03). From S3 on, **one commit per finished task**,
made when the task is done (the user asked for this). There is no remote, so CI has still never run.

**N3 data (2026-11-01).** `scripts/get_data.sh [tinystories|wikitext|enwik9|all]` + `scripts/slice_data.py`.
- In `data/` (gitignored): `tinystories.{10M,100M,1G}.txt` (no `.full`; the 2.2 GB raw file is in `data/raw/`) and
  `wikitext.{10M,100M,full}.txt` (full = 540 MB). `data/CHECKSUMS.sha256` verifies with `cd data && shasum -a 256 -c CHECKSUMS.sha256`.
- Verified: a rerun gives identical checksums, every output decodes as strict UTF-8, and the document splitters were
  tested at every chunk boundary.
- WikiText: the TASKS title rule `^ = [^=].* = $` gives 29,444 documents (the official article count is ~28,475; a few
  in-article lines match too). That is deterministic and fine for exactness. The 10 MB slice has 850 distinct non-ASCII characters.
- enwik9 is implemented but **not downloaded or run** (1.3 GB of disk). Disk is now ~24 GB free.
- `bpe_ref.py --words-only` takes under 1 s per 10 MB slice; outputs are in `runs/ref/{tinystories,wikitext}.10M/words.tsv`.

**S1 (2026-11-02).** `include/parabpe/regex.hpp` + `src/regex.cpp` (`Regex`, `Matcher::match_at` / `for_each_match`, which
throw on PCRE2 errors, gaps and empty matches, with a 16 MB JIT stack); `validate_utf8` + `make_work_units` in `corpus.cpp`;
`SerialPreTokenizer`; `write_words`; CLI `--special` (repeatable), `--no-special`, `--dump-words` (stops after phase 1).
- Tests: `test_chunker.cpp`, `test_pretok.cpp` (includes the golden `words.tsv` check for all 11 cases),
  `test_unicode_parity.cpp` (all code points × `\s`, `\p{L}`, `\p{N}` against `tests/golden/unicode_classes.txt`), and the
  shared `tests/unit/golden_case.hpp` (reads `case.txt`; reuse it in S2). **34/34 pass** (Release), zero warnings.
- **Unicode finding:** PCRE2's `\s` matched U+180E (Mongolian vowel separator); Unicode 16 and Python don't. Fixed in C++
  by compiling `\s`/`\S` as `\p{White_Space}`/`\P{White_Space}` (no SPEC change). U+180E is now in `whitespace_zoo`.
- Acceptance: `parabpe train --dump-words` gives byte-identical `words.tsv` to the golden files for all 11 cases (each with
  its own `--special` flags) and to `bpe_ref.py --words-only` on `tinystories.10M` and `wikitext.10M`.
- Serial phase 1 speed: ~105 MB/s (1 GB TinyStories: pretok 9.6 s, chunk + UTF-8 check ~1.5 s, 45k unique words).
- **Sanitizers:** ASan **cannot run on this Mac** (see Environment notes). UBSan-only build: 34/34 pass, and CLI runs are clean.
  The ASan check for S1 comes from the CI `asan` job (not yet run, because nothing is pushed).

**S2 (2026-11-02).** `trainer.{hpp,cpp}` (`count_words` / `train_text` / `train`), `word_table.cpp`, NaiveSerial in
`merge_naive.cpp`, `io.cpp` (GPT-2 `bytes_to_unicode`, `build_vocab`, byte-exact `format_merges` / `format_vocab`,
`format_timings` with sysctl or /proc/cpuinfo hardware info + git SHA), and `src/bpe_common.hpp`, the shared SPEC §3/§4
rules that **every engine must use**. `MergeEngine::run` now takes `PhaseTimings&` (fills `pairs` and `merge`).
- Tests: `test_golden.cpp` (merges.txt + vocab.json byte-exact for all 11 cases; loops over `kEngines`) and
  `test_io.cpp`. **46/46 pass** (Release and UBSan), zero warnings. A mutation check (flipping the tie-break)
  makes 10/11 golden cases fail, so the test has teeth.
- **Real data:** C++ NaiveSerial and `bpe_ref.py` give **identical merges.txt + vocab.json** on `tinystories.10M` and
  `wikitext.10M` at vocab 1000 (743 merges each). Python takes 10 s / 74 s; C++ takes 0.27 s on the merge phase for TinyStories.
- `tests/interop/check_hf.py --model DIR --heldout FILE`: HF `tokenizers` ids == `bpe_ref.encode` ids. It passes on
  `runs/s2/ts10M_v1000` against 1 MB of WikiText (494k tokens).
- `timings.json` on this Mac reports `performance_cores: 5, efficiency_cores: 6`; `git_sha` is `"unknown"` until the first commit.
- Code in the new and rewritten files is formatted with Xcode's clang-format
  (`{BasedOnStyle: Google, ColumnLimit: 100, AllowShortCaseLabelsOnASingleLine: true, Standard: c++20, SortIncludes: Never, AllowShortFunctionsOnASingleLine: Inline}`).

**S3 (2026-11-03).** IncrSerial in `src/merge_incr.cpp`: pair counts + inverted index (pair → word ids, consecutive
duplicates dropped) + a lazy max-heap ordered by `better_pair`, whole-word diff per affected word, `last_touched` stamps,
heap rebuild above 4× live pairs. Because a merge creates a fresh id `z`, the pairs new to a word are exactly the ones
containing `z`, so only those are appended to the index.
- **Bug found and fixed while building it:** pairs containing `z` were pushed onto the heap before `token_bytes[z]`
  existed, so the comparator read past the end of the vector (wrong tie-breaks, 9/11 golden cases failed). UBSan does
  not see this. The Debug build now turns on libc++/libstdc++ bounds checks (`_LIBCPP_HARDENING_MODE_EXTENSIVE`,
  `_GLIBCXX_ASSERTIONS`); with the bug put back, the Debug tests stop with SIGTRAP (exit 133).
- Tests: `kEngines` in `test_golden.cpp` now includes IncrSerial (byte-exact on all 11 cases); `test_io.cpp` runs the
  early-stop and empty-corpus tests for both serial engines and compares IncrSerial with NaiveSerial on 200 random
  small-alphabet corpora (many count ties, long runs).
- `tests/equivalence/run_matrix.py --input F --vocab-size N [--threads ...] [--binary ...]`: every implemented
  (pretok, merge) pair × threads {1,2,3,4,8,16} against the oracle (serial + NaiveSerial at 1 thread), comparing sha256 of
  `merges.txt` and `vocab.json`. Strategies that throw "not implemented" are listed as SKIP. Exit 0 / 1 mismatch / 77 no input.
  Registered in CTest as `equivalence_matrix` (label `slow`, tinystories.10M at vocab 4000): 12/12 identical, 6 pairs skipped.
- **Real data:** IncrSerial == NaiveSerial (merges.txt + vocab.json) on `tinystories.10M` and `wikitext.10M` at vocab 1000
  and 4000, and == `bpe_ref.py` at vocab 1000. Merge phase at vocab 4000: TinyStories 1.087 s → 0.0087 s (**125×**),
  WikiText 12.74 s → 0.048 s (**266×**); the card asked for ≥10×. Outputs are in `runs/s3/`.

**S4 (2026-11-03).** `src/pretok_omp.cpp`: one `parallel` region, a `Matcher` per thread, `omp for schedule(dynamic,1)`
over work units into per-thread maps pre-bucketed by **low** hash bits (B = T rounded up to a power of 2); after the
barrier, `omp for` over buckets merges bucket b of every thread into shard b (no locks); shards are concatenated and
sorted. Per-thread and per-shard structs are `alignas(64)`. Exceptions inside the region are captured and rethrown.
- New config `TrainConfig::pretok_reduce` (`--pretok-reduce bucketed|critical`): `critical` merges one map per thread
  into a global map in a critical section, the report's comparison. `timings.json` `config` gained `"pretok_reduce"`
  (additive, like `performance_cores`; SPEC §6 example not changed).
- **Perf bug found:** bucketing by the HIGH hash bits (as the card said) made 2 threads 6.6× and 3 threads 51× slower than
  serial, because `ankerl` also indexes by the high bits. Fixed by using the low bits; card and PLAN updated.
- Tests: golden `words.tsv` for OMP × threads {1,2,3,4,8,16} × {bucketed, critical} with 16-byte work units; OMP vs
  serial on a 20k-piece many-document text; error propagation from a worker thread. Debug (bounds-checked) build passes.
- Acceptance: `words.tsv` identical to serial on both 10 MB slices (threads {1,2,3,4,8}, both variants), on
  `tinystories.1G` (threads {1,2,3,4,5,6,8,11}, both variants) and on `wikitext.full` (threads {2,5,11}).
- **Phase-1 scaling, tinystories.1G** (serial 9.58 s; one run each, not yet the bench methodology):
  | threads | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 11 |
  |---|---|---|---|---|---|---|---|---|
  | bucketed (s) | 9.42 | 4.70 | 3.23 | 2.42 | 2.19 | 1.94 | 1.67 | 1.37 |
  | speedup vs serial | 1.02 | 2.04 | 2.97 | 3.96 | 4.37 | 4.94 | 5.75 | 7.01 |

  Bucketed and critical are within noise here (only 45k unique words). On `wikitext.full` (540 MB, 599k unique words):
  serial 5.42 s; 11 threads bucketed 1.05 s vs critical 1.34 s; 5 threads 1.47 s vs 1.52 s.
- Full pipeline, tinystories.1G, 11 threads, `--pretok omp`, vocab 1000: total 2.19 s, of which **chunk 0.75 s** (serial
  mmap + UTF-8 check + special-token split) and pretok 1.42 s. Chunking is now the Amdahl bottleneck: a G1 item.
- **TSan:** CI job `tsan` (clang + libomp-dev, Archer via `OMP_TOOL_LIBRARIES`, `ctest -LE slow`, `OMP_NUM_THREADS=4`)
  is written but **has never run** (no remote). Locally, Apple clang's TSan fails at startup ("Interceptors are not
  working", also with `DYLD_INSERT_LIBRARIES`), same toolchain problem as ASan.

**Next task:** N8 (profiling spike), then N7 (bench harness), then G1 (profiling gate: a user decision on S6 effort).

## What exists
| Path | What it is |
|---|---|
| `SPEC.md` | Frozen training spec + file formats (§6). This is the contract. |
| `docs/PLAN.md` | Architecture, rationale, benchmark methodology, timeline (reviewed by an independent agent; v2) |
| `docs/TASKS.md` | Task cards with acceptance criteria |
| `include/parabpe/types.hpp` | `TokenId`, `PairKey`, `Merge`, `TrainConfig`, `WordCounts`, `WordTable`, `PhaseTimings`, `Vocab`, `kGPT2Pattern` |
| `include/parabpe/corpus.hpp` + `src/corpus.cpp` | `Corpus` (mmap, **implemented**), `WorkUnit`, `make_work_units` (stub, S1) |
| `include/parabpe/pretokenizer.hpp` | `PreTokenizer` interface + `make_pretokenizer` factory |
| `include/parabpe/merge_engine.hpp` | `MergeEngine` interface, `build_word_table`, `make_merge_engine` |
| `include/parabpe/io.hpp` + `src/io.cpp` | `bytes_to_unicode`, `build_vocab`, writers (all stubs, S2) |
| `src/strategies.{hpp,cpp}` | Internal per-strategy constructors + factory dispatch |
| `src/pretok_{serial,omp}.cpp`, `src/merge_{naive,incr}.cpp`, `src/word_table.cpp` | Strategies; NaivePar (S7) and IncrPar (S6) paths still throw |
| `src/main.cpp` | CLI11 `train` / `encode` subcommands, wired to the full pipeline with phase timing |
| `CMakeLists.txt` | Targets `parabpe_core` (lib), `parabpe` (CLI), `parabpe_tests` (gtest). Options `PARABPE_NATIVE/ASAN/TSAN` |
| `.github/workflows/ci.yml` | GCC Release, ASan and TSan (clang + LLVM libomp) jobs on ubuntu |
| `requirements.txt` | Python deps for the reference, bench and interop scripts |
| `reference/` | `bpe_ref.py` (oracle), `make_golden.py`, `test_bpe_ref.py`, `README.md` (versions + byte formats) |
| `tests/golden/` | 11 cases (`input.txt`, `case.txt`, `words.tsv`, `merges.txt`, `vocab.json`) + `unicode_classes.txt` |
| `scripts/build_pcre2.sh` | Builds PCRE2 10.45 (static) for CI |
| `scripts/get_data.sh`, `scripts/slice_data.py` | Corpus download + normalization/slicing (N3) |
| `include/parabpe/regex.hpp`, `src/regex.cpp` | PCRE2 wrapper: `Regex` (shared, JIT) + per-thread `Matcher` (S1) |
| `include/parabpe/trainer.hpp`, `src/trainer.cpp` | The pipeline as a library: `count_words`, `train_text`, `train` (S2) |
| `src/bpe_common.hpp` | Shared merge rules for every engine: tie-break, greedy merge, merge count (S2) |
| `tests/interop/check_hf.py` | HF `tokenizers` token-id check of our output files (S2) |
| `tests/equivalence/run_matrix.py` | Equivalence matrix (S3); CTest `equivalence_matrix`, label `slow` |
| `bench/`, `report/` | Empty (contain only `.gitkeep`) |

## Environment notes (developer Mac)
- macOS on Apple Silicon (arm64). The default C++ compiler is **Apple clang 17** (`/usr/bin/c++`).
  The `clang` on PATH is the **swiftly** toolchain, which CMake does not use.
- `cmake` 3.29 is installed from **MacPorts** (`/opt/local/bin`). `libomp` 22 and `pcre2` 10.45 come from **Homebrew**.
- **OpenMP on macOS:** `CMakeLists.txt` asks `brew --prefix libomp` for the install location and sets `OpenMP_CXX_FLAGS`, so `find_package(OpenMP)`
  works with Apple clang. Don't remove that block.
- PCRE2 is found with `find_path`/`find_library` plus Homebrew hints, because `pkg-config` is **not installed**.
- Homebrew `llvm` is **not installed**. Apple clang's TSan fails at startup on macOS 27 ("Interceptors are not
  working"); `brew install llvm` might give working ASan/TSan runtimes (untried). Until then TSan is CI-only.
- **ASan is broken on this Mac (2026-11-02):** Apple clang 17 (SDK 15.5) on macOS 27 deadlocks in the ASan runtime's own
  startup. Even a hello-world built with `-fsanitize=address` hangs, so `-DPARABPE_ASAN=ON` builds hang during test
  discovery. Xcode 16.4 is installed and active and has the same clang, so it doesn't help; an Xcode that supports
  macOS 27 (or `brew install llvm`) may fix it. Until then use UBSan locally:
  `cmake -B build-ubsan -DCMAKE_BUILD_TYPE=Debug -DPARABPE_NATIVE=OFF -DCMAKE_CXX_FLAGS="-fsanitize=undefined -fno-sanitize-recover=undefined" -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=undefined`
  and rely on the CI `asan` job for ASan.
- **Python:** the repo venv `.venv` exists (Python 3.13.3, `requirements.txt` installed). Always use
  `.venv/bin/python`: the system Python has no `regex`, and other `regex` versions break the Unicode pin.
- The first configure downloads GoogleTest v1.15.2, CLI11 v2.4.2 and unordered_dense v4.4.0 with FetchContent, so it needs network access.

## Benchmark machine (decided 2026-10-31)
The user decided the 16-core requirement was ambitious, not a course rule. **Final benchmarks run on this Mac.**
A 16-core machine is an optional extra (e.g. an AWS `c7a.4xlarge` or `c7g.4xlarge` = 16 physical cores, no SMT,
for a few hours at the end).
- Hardware: **Apple M3 Pro, 5 performance + 6 efficiency cores, 18 GB RAM**, no SMT.
- Measured with a compute-bound OpenMP toy loop (2026-10-31, best of 3, `schedule(dynamic,1)`):

  | threads | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 11 | 16 |
  |---|---|---|---|---|---|---|---|---|---|
  | speedup | 1.00 | 1.96 | 2.80 | 3.46 | 4.15 | 4.66 | 5.21 | 6.30 | 7.65 |

  Close to linear up to 5 (P-cores), then each E-core adds ~0.4 of a P-core. The ideal ceiling is ≈ 7.4
  "P-core equivalents". `schedule(static)` was slightly worse (6.14× at 11). Memory-bound phases will scale less.
- **No thread pinning on macOS:** with `OMP_PROC_BIND=close OMP_PLACES=cores`, `omp_get_num_places()` returns 0.
  The bench harness can still set those variables (harmless here, useful on Linux).
- **No `perf`.** Profile with Instruments (`xctrace record --template 'Time Profiler'`).
- Benchmark hygiene on the laptop: on mains power, nothing else running, lid open/cool. Record this in the report.
- Report framing: threads {1,2,3,4,5} as the P-core region and {6,8,11} as the heterogeneous region. Phases with
  static owner partitioning (S5, S6) are gated by the slowest core; that is a finding, not a bug.
- The equivalence matrix still uses threads {1,2,3,4,8,16}. Oversubscription (16 on 11 cores) is fine for correctness.

## Decisions already made (don't re-litigate)
- C++20 + OpenMP. `std::thread` and MPI are optional, only if ahead of schedule.
- PCRE2, not RE2 (RE2 lacks the lookahead the pattern needs). `ankerl::unordered_dense` instead of `std::unordered_map`.
- **IncrSerial is the official speedup baseline.** NaiveSerial and NaivePar run only on small configurations.
- Chunking happens only at special-token (document) boundaries, never at `\n\n` (SPEC §2).
- Tie-break on the tuple `(bytes(a), bytes(b))` (SPEC §4).
- TSan needs clang + LLVM libomp. GCC's libgomp reports false races.
- Expected IncrPar speedup is 1–2×. The report's centrepiece is the per-phase Amdahl story.

## Open questions for the user / team
- The team still needs to review `SPEC.md` and the interfaces (N2). The current text is a draft written by the agent.
- Team size and who owns which task (`docs/PLAN.md` §7 suggests a split for 3 people).
- Big-machine access (N5) is optional now (see "Benchmark machine"). Not arranged.
- Submission deadline: not recorded. Add it here once known.
