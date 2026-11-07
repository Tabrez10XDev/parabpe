# Handoff: read this first

## Current state (2026-11-07)
**Done:** N1 (repo skeleton), N2 (draft spec and interfaces), **N4 (Python reference + golden files)**, **N3 (data)**, **S1 (chunker + serial pre-tokenizer)** and **S2 (WordTable + NaiveSerial +
writers + `train()`)** **S3 (IncrSerial + equivalence harness)** and **S4 (OMP pre-tokenizer + TSan CI job)**. Training works end to end
with `--pretok serial|omp --merge naive-serial|incr-serial`; All four engines work: IncrPar has a parallel phase 2 (S5) and phase 3 (S6).

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

**N8 (2026-11-03).** `scripts/profile_phases.py` (run with `.venv/bin/python -I`; `--small`, `--large`, `--threads`,
`--out`) times the `bpe_ref.py` phases and compares HF `tokenizers` training at Rayon 1/N with our C++.
Results and takeaways are in **`docs/notes/profiling.md`** (raw output in `runs/n8/`). In short:
- In C++, phase 1 is 76–95% of serial time.
- The merge loop is 34% of 11-thread wall time on `wikitext.full`, but 3% on TinyStories.
- A cold page cache makes `chunk` 10× slower on 1 GB.
- We beat HF by about 11× (1 thread) and 8× (11 threads) on `wikitext.100M`. This is ballpark only: HF doesn't split
  on specials.
- The spike ran on battery. S8 must run on mains power.

**N7 (2026-11-03).** The bench harness:
- `bench/run_bench.py <config.yaml>` (run with `.venv/bin/python -I`; options `--reps`, `--results-dir`, `--require-ac`,
  `--dry-run`) appends one CSV row per run to `results/<config>.csv`.
- Each configuration gets a warm-up run (`warmup=1`).
- The runner sets `OMP_PROC_BIND=close OMP_PLACES=cores` and uses `numactl` on multi-socket Linux.
- Configurations that only use serial strategies run at 1 thread only. "Not implemented" strategies are skipped.
- Every row records the `merges.txt` sha256, and the runner flags any mismatch within an (input, vocab) pair. The
  benchmark therefore doubles as an equivalence check.
- Every row also records the power source and the child's CPU time (`cpu_s`).
- Phase columns are `<phase>_s`, because `pretok` and `merge` are also strategy columns.

Configs in `bench/configs/`:
- `smoke` (about 30 s)
- `main` (the PLAN §6 grid, 100 configurations)
- `naive` (10 MB, vocab 4k)
- `reduce` (bucketed vs critical)
- `threshold` (IncrPar sweep)
- `weak` (50 MB per thread, zipped; make the slices first, the command is in the file)

`bench/plot.py results/*.csv --out-dir results/plots` writes:
- `speedup_<input>_v<vocab>.png`: per-phase speedup vs IncrSerial, with a median line, a min/max band, an ideal
  line and a shaded E-core region.
- `phases_*.png`: stacked bars of median phase times.
- `amdahl.png`: serial fraction fitted on threads 1–5.
- `weak_scaling.png` and `threshold.png`.
- `summary.md`: the table view.

Warm-up runs are excluded. Runs with wall time > 2 × CPU time are dropped as stalls and counted.
The plots use the dataviz reference palette in fixed slot order (validated: CVD ΔE ≥ 9.1 adjacent).
Checked with `smoke` (real runs) and a synthetic CSV for the weak and threshold plots.
- **Finding:** on battery with the user away, the Mac sleeps mid-run even with `caffeinate -i` (`-s` only works on
  AC). One 0.1 s run read 295 s, and two naive runs read 162 s and 553 s for 13 s of CPU (`pmset -g log` confirms
  Sleep/DarkWake). Runs during dark wake are also throttled: serial pretok on 10 MB took 0.22 s instead of 0.09 s.
  **S8 needs the Mac on mains power and awake.** Use `--require-ac`.

**S7 (2026-11-03).** NaivePar in `src/merge_naive.cpp`: each merge recounts all pairs in parallel. Words are counted
under `omp for schedule(dynamic,1024)` into per-thread maps, pre-bucketed by the low hash bits. Bucket b of every
thread is reduced into shard b, which also finds its own best pair. The master then takes the best of the shard
winners: `better_pair` is a total order on distinct pairs, so the result does not depend on the partition. The merge
is applied with `parallel for` over words (each word is a disjoint range of `tokens`).
- Tests: the golden test now runs parallel engines at threads {1,2,3,4,8,16}. The random-corpus test checks NaivePar
  at 1/3/4/16 threads against NaiveSerial. The early-stop and empty-corpus tests cover every engine.
  54/54 pass (Release and Debug with bounds checks).
- Equivalence matrix: 0 mismatches on `tinystories.10M` (vocab 4000) and `wikitext.10M` (vocab 2000, 36/36).
- Not timed yet: this session ran on battery with the Mac sleeping and throttling. NaivePar timings come from the
  `naive` bench config in S8.

**S5 (2026-11-03).** `src/merge_incr.cpp` is one `IncrEngine` for both strategies:
- Words are split into **owners**, contiguous ranges balanced by token count: 1 owner for IncrSerial, `cfg.threads`
  for IncrPar. Each owner has a private inverted index.
- Global counts live in `ShardedCounts`: power-of-two shards chosen by the low hash bits. With one shard (serial),
  no hash is computed.
- Parallel phase 2: `omp for schedule(dynamic,1)` over owners builds each private index and counts into per-thread
  maps pre-bucketed by shard. Shard s then sums bucket s of every thread, with no locks.
- Phase 3 (shared, still serial; `TODO(S6)`) walks every owner's list for the chosen pair. New pairs go to the
  owner's index. A pair whose count reaches 0 is erased from every owner.
- Tests: IncrPar is in the golden test at threads {1,2,3,4,8,16} and in the random-corpus test at 1/2/3/8/16 threads
  (owners can be empty there). 56/56 pass (Release and Debug).
- Equivalence matrix on `tinystories.10M` v4000: 0 mismatches, **no strategy skipped any more**.
- `wikitext.full`, vocab 32000 (one run each, on battery):
  - Phase 2 drops from 0.032 s (IncrSerial) to 0.015 s at 5 threads and 0.012 s at 11.
  - Phase 3 under IncrPar is **13–19% slower** than IncrSerial (0.75–0.79 s vs 0.66 s): each merge looks up every
    owner's index. S6 has to win this back.
  - `merges.txt` is identical to IncrSerial at 5 and 11 threads.

**G1 (2026-11-03).** The numbers and the decision are in `docs/notes/profiling.md`. The user chose a **modest S6**
(as planned, reported honestly) plus a new task, **S10 (parallel chunker)**; its card is in TASKS.

**S6 (2026-11-03).** IncrPar phase 3, in `src/merge_incr.cpp`:
- The master pops the heap and runs the loop. A step whose **tokens touched** reach `cfg.par_threshold_tokens`
  (checked by counting listed words first, then summing their lengths) opens a parallel region with three steps:
  1. `omp for` over owners: `merge_owner` rewrites the owner's words and buffers deltas in a padded per-thread map,
     bucketed by counts shard.
  2. `omp for` over shards: sum the deltas over all threads **before** applying them, so a count never passes through
     a transient 0. Then apply them and collect heap pushes and zeros.
  3. `omp for` over owners: drop the zero pairs from each index.

  The master then pushes to the heap. Smaller steps run on the master alone, through the same `merge_owner` code
  path. Negative counts are flagged inside the region and thrown by the master.
- **Deviation from the card:** I used no persistent region. With about 31k merges, most touching a few words, a
  barrier on every merge would cost more than it saves; forking only for big steps costs nothing on small ones.
  Mention this in the report.
- Tests: the golden and random-corpus tests run IncrPar with threshold 0, so every step is parallel; the random test
  also uses threshold 8, a mix of both paths. A mutation (dropping thread 0's deltas) fails both tests.
  56/56 pass (Release and Debug).
- Equivalence matrix: 0 mismatches on `tinystories.10M`, with every strategy implemented.
- `wikitext.full`, vocab 32000, phase 3 (one run each, **on battery**, indicative only). `merges.txt` was identical
  in every run.

  | | 5 threads | 11 threads |
  |---|---|---|
  | IncrSerial | 0.675 s | 0.675 s |
  | IncrPar, threshold 0 (every step parallel) | 2.28 s | 5.13 s |
  | IncrPar, threshold 1024 | 0.62 s | 0.84 s |
  | IncrPar, threshold 4096 (default) | **0.594 s (1.14×)** | 0.698 s |
  | IncrPar, threshold 16384 | 0.63 s | 0.653 s (1.03×) |
  | IncrPar, threshold 65536 | 0.68 s | 0.667 s |

  This is a modest result, as G1 expected: S6 wins back S5's per-owner cost and gains about 1.1× at 5 threads.
  At 11 threads, the E-cores and the extra owner lookups on every serial step cancel the gain. `threshold.yaml`
  sweeps this properly in S8.

**S10 (2026-11-03).** The parallel chunker, in `src/corpus.cpp`. `make_work_units(..., threads)` and
`validate_utf8(text, threads)` take a thread count. `train()` passes `cfg.threads` with `--pretok omp` and 1 with
`--pretok serial`, so the serial baseline keeps the old code. Texts under 1 MB stay serial.
- **UTF-8 check:** T byte ranges, each moved forward past continuation bytes, with a `reduction(min)` over each
  range's first error. In valid UTF-8 every non-continuation byte starts a sequence, so each range agrees with the
  serial scan up to the first error: the reported offset is the same for every T.
- **Special split:** each thread collects every occurrence (overlapping ones included) that **starts** in its range,
  searching up to `end + longest − 1`, sorted by position with the longest first. One serial greedy pass skips
  occurrences before `pos`, which is the same leftmost/longest rule as the serial scan. Packing into units is the
  shared `UnitPacker`.
- Tests (`test_chunker.cpp`): units identical to serial for threads {1,2,3,4,8,16} on a 3 MB random text, with
  5 special-token sets (including self-overlapping `aa` and the prefix pair `<|end` / `<|endoftext|>`) and two unit
  sizes; the same error offset for every T with invalid bytes at, before and after every range border. A mutation
  (no boundary alignment) fails both tests. 58/58 pass (Release and Debug).
- `words.tsv` identical to serial on `tinystories.1G` and `wikitext.full` (5 and 11 threads). Matrix: 0 mismatches.
- Warm `chunk` on `tinystories.1G`: 0.150 s at 1 thread, 0.058 s at 5, 0.051 s at 11 (2.9×; memory-bandwidth bound).
  The whole 1 GB run at 11 threads (vocab 1000) is **1.47 s**, down from 1.66 s. One run each, on battery.

**B0 (2026-11-03).** The batched-merging upgrade is planned in `docs/notes/batched_merging.md`, which has the full
proof. Task order from here: B1, B2, B3, B4, B5, N6, S8, B6, B7, S9 (cards in `docs/TASKS.md`).
- **The formula needs a 0-based j.** With 1-based members, `j+1` keeps one member too many. Counterexample in the note:
  words `eax`×2 and `df`×2.
- `tools/batchsim/batchsim.cpp` (standalone, not in CMake): `c++ -std=c++20 -O2 tools/batchsim/batchsim.cpp -o build/batchsim`.
  Its merges are byte-identical to IncrSerial on `wikitext.10M` v4000.
- All six prototype round counts reproduce exactly under the relaxed rule. The closed form is better: **2,561 rounds**
  on `wikitext.full` v32k (mean batch 12.39) against 3,409, and 21,295 against 32,533 at v100k.
- **`batch_scan_cap=256` costs real batching at v100k:** 35,237 rounds against 21,295 uncapped; 1024 gives 21,646. B3
  settles the default.
- Not reproduced: hashed imbalance is 1.07–1.08 (prototype 1.02–1.06) and the batched big-step share is 90.0% (prototype
  91.2%). The contiguous 2.30 and the 80.5% do reproduce.
- **Wrong rules pass on `wikitext.full` v32k.** `all` and `one-based` only fail on smaller corpora, so a clean verify on
  one big corpus proves little.

**CI (2026-11-03).** The repo now has a remote (`origin`, public), so CI does run; the notes above saying otherwise
predate it. First run: `gcc` and `asan` passed, `tsan` failed 9/58 on false positives inside `libomp.so`, stopping each
test at the first report, so TSan had checked nothing. B0 adds `ignore_noninstrumented_modules=1`. Check
`gh run list` after each push.

**B1 (2026-11-03).** The shared incremental-engine pieces moved from `src/merge_incr.cpp` to `src/incr_common.hpp`, no
logic changes. `parallel_step` is a free template taking `merge(table, owner, add)`. 58/58 in Release and Debug; matrix 0
mismatches on both 10 MB slices. Old vs new binary, alternating, median of 7 on `wikitext.full` v32k (battery, indicative):
IncrSerial 0.704 vs 0.691 s, IncrPar at 5 threads 0.588 vs 0.585 s; `merges.txt` identical.

**B2 (2026-11-03).** `incr-batch` is wired through the CLI, `name_of` and the factory; `src/merge_batch.cpp` is a stub
(`TODO(B3)`). New `TrainConfig` fields, `MergeStats` in `TrainResult`, `MergeEngine::stats()`, and a `"batch"` block in
`timings.json` for incr-batch only (additive, like `pretok_reduce`; the SPEC §6 example is unchanged). `--trace` is
accepted but does nothing until B6.

**B3 (2026-11-05).** `src/merge_batch.cpp` (engine, lazy-heap `select_round`, `SequentialOracle` for `--verify-batches`)
and `src/batch_select.hpp` (pure `choose_batch`, `keep_count`, `Batch`, `apply_batch`). The engine is still serial: one
owner, threads ignored. `tests/unit/test_batch.cpp` is new; `incr-batch` is in the golden test, the `Engines` suite, the
random-corpus list (oracle on) and `run_matrix.py`.
- 77/77 in Release and Debug. Matrix 0 mismatches on both 10 MB slices with `incr-batch` included.
- Engine vs batchsim, oracle on: **0 disagreements** on rounds, max batch and `merges.txt` over 8 configurations, including
  `wikitext.full` v32k (**2,561 rounds**, target ≤ 3,409) and v100k (21,295).
- All three rules are exact on 200 random corpora (`make_batch_engine(BatchRule)`; only Closed ships).
- **Mutations, all caught:** always k=K; 1-based j; self-pair mid-batch; `<=` in g; stopping pair left out of T; incomplete
  scan treated as complete; tie scan skipped; bytes pushed after the heap push (Debug trap, like the S3 bug).
- **Deviations from the card:** `batch_scan_cap` defaults to **4096**, not 256, because of the round counts above. The tie
  scan stops early once k reaches max(g,1), which cuts selection time and returns the same batch.
- **Serial cost:** selection dominates on tie-heavy data. See the note's measurements. This is the main risk for B4's
  kill check at v100k.
- CI: B0, B1 and B2 runs are fully green, TSan included, so the existing parallel code has now been race-checked.

**B4 (done 2026-11-07; code 2026-11-05).** Kill check passed on mains power; see the sweep below.
- `partition_words_hashed` (`Owner::words`), rounds at or above `par_threshold_tokens` through the templated
  `parallel_step`, and a periodic parallel sweep of zero pairs (every 16k pairs). IncrSerial and IncrPar keep their paths.
- **Added beyond the card: `batch_skip_small`** (default on; `--batch-full-rounds` turns it off). Rounds that can't go
  parallel merge one pair. It took v100k from 0.79× to 1.41×. Span figures must come from `--batch-full-rounds` runs, which
  are the ones that match batchsim. Details and battery numbers are in `docs/notes/batched_merging.md`.
- 81/81 in Release and Debug. CI green, TSan included: **the first race check of parallel rounds.** Matrix 0 mismatches on
  both 10 MB slices.
- `--verify-batches` passes at 5 and 11 threads at v32k and v100k, with and without skipping; `merges.txt` equals IncrSerial
  in all eight runs.
- Mutations: 16 of 17 caught. The survivor (no per-round stamp) is an equivalent mutant, explained in the note.
- Two stale `build-asan/parabpe_tests --gtest_list_tests` processes (the ASan deadlock) ran alongside all the B4 battery
  timings at about 45% of a core each. They had gone by 2026-11-06.
- The Mac slept during a foreground test run even in the middle of a session. Wrap long runs in `caffeinate -i`.

### B5 (2026-11-06, done out of order: B4's kill check is still pending)
- `--owner-map contiguous|hashed` for IncrPar, default contiguous. `--owner-stats` adds the load measurement; it is off by
  default because it reads each touched word again, which would skew IncrPar timings.
- Real engine, `wikitext.full` v32k, 572 big steps: busiest/mean 2.30 → 1.08 (T=5), 4.02 → 1.18 (T=11). Same step count
  and ratios as batchsim, which shares no code with it. `merges.txt` equals IncrSerial in all four runs.
- **Result worth reporting: the imbalance fix alone barely helps.** Phase 3 vs IncrSerial, median of 5, battery, indicative:

| | T=5 contiguous | T=5 hashed | T=11 contiguous | T=11 hashed |
|---|---:|---:|---:|---:|
| v32k | 1.15× | 1.23× | 0.97× | 0.99× |
| v100k | 1.11× | 1.14× | 0.94× | 0.95× |

- Off-path cost: IncrSerial 0.602 s vs 0.597 s before B5, IncrPar 0.503 s vs 0.520 s, within noise.
- 83/83 in Release and Debug. Matrix 0 mismatches on both slices, with `incr-par:hashed` as a new column.
- Mutations: 4 of 4 caught, after one survivor showed the test never ran a small step; the test now does.
- The two stale ASan processes have gone (not killed by us). IncrSerial v32k now takes about 0.59 s on battery, against
  0.689 s when they were running, so B4's battery figures carried that load and the kill check must not reuse them.

### N6 (2026-11-06)
`docs/notes/related_work.md`. **BatchBPE already merges many pairs per batch**, and HF and YouTokenToMe already
parallelise the merge step, so the report may not say "first batched" or "first parallel". What we did not find: several
merges per parallel round with a proof that the merge list, order included, is byte-identical to sequential training.
BatchBPE's README says its merge order differs. Use the wording in the note. S9 follow-ups are on the card.

### S8 tooling (2026-11-07, no benchmark results yet)
- `run_bench.py`: incr-batch, `owner_map`, `owners_per_thread`, `full_rounds`, `owner_stats`, `baseline`, round and load
  columns. Power is read before every run. Stalled runs are repeated. The CSV header changed; no old results existed.
- `plot.py`: every row has a `kind`, and only `main` rows reach headline charts, replacing a hardcoded threshold of 4096
  that would have hidden incr-batch. New charts: span, batch sizes, owner maps. End labels no longer collide.
- `bench/kill_check.py` prints the sweep and verdict, and exits 2 rather than give a verdict on battery data.
- `scripts/run_s8.sh` runs everything after the sweep. Estimate about 2¼ hours, plan for 3.
- Tested end to end on the 10 MB slices only. Those numbers were throwaway and are not in the repo.
- **The Mac idle-slept on battery during timing runs even under `caffeinate -i`** (`Maintenance Sleep` / `Idle Sleep` in
  `pmset -g log`): 6 s runs took up to 15 minutes of wall time. Not yet seen on mains, where `run_bench.py` adds `-s`.
  Check the "stalled runs" count at the end of each config.
- Palette: 8 slots pass. incr-par vs incr-batch is ΔE 6.9 (deutan), legal with their different marker shapes.

### B4 kill check (2026-11-07, mains power)
`bench/configs/b4_sweep.yaml` on `wikitext.full`: 156 rows, 0 stalled, 0 failed. `bench/kill_check.py`: **PASS**.
Phase 3 speedup vs IncrSerial, `owners_per_thread=1`, median of 3:

| Threads, vocab | 1024 | **4096** | 16384 | 65536 |
|---|---:|---:|---:|---:|
| 5, v32k | 1.92× | **1.83×** | 1.54× | 1.32× |
| 5, v100k | 1.37× | **1.39×** | 1.27× | 1.15× |
| 11, v32k | 1.44× | **1.47×** | 1.33× | 1.17× |
| 11, v100k | 1.08× | **1.20×** | 1.14× | 1.06× |

- **Defaults set:** `owners_per_thread` 4 → **1** (it beat 2 and 4 in every cell). The threshold stays **4096**, shared
  with IncrPar: 1024 wins only at 5 threads v32k. The user approved both.
- `--verify-batches` passes with the new default at 5 and 11 threads, v32k and v100k; `merges.txt` equals IncrSerial in all four.
- **11 threads is slower than 5 in every cell**: the E-cores hold each round back. Report P-core and E-core regions apart.
- Noise: the largest min-max spread is 22% (incr-batch) and 12% (IncrSerial, v32k t5); others 0–2% for IncrSerial.
- **Baseline guard:** IncrSerial phase 3 is 0.61–0.64 s at v32k, below the plan's 0.66–0.68 s. The old figure was taken
  while the two stale ASan processes ran, so this is the cleaner baseline, not a regression. S8 re-measures it anyway.

**Next task:** S8 on mains power: `scripts/run_s8.sh` (about 2¼–3 h, lid open), check the stalled counts, then
`scripts/run_s8.sh --clean`. S8 **needs the Mac on mains power and
awake**: `.venv/bin/python -I bench/run_bench.py bench/configs/<name>.yaml --require-ac`, then
`bench/plot.py results/*.csv --out-dir results/plots`.

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
| `src/pretok_{serial,omp}.cpp`, `src/merge_{naive,incr}.cpp`, `src/word_table.cpp` | Strategies; IncrPar's phase 3 is still serial (S6) |
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
| `bench/run_bench.py`, `bench/plot.py`, `bench/configs/*.yaml` | Bench harness (N7); results go to `results/` (committed) |
| `scripts/profile_phases.py`, `docs/notes/profiling.md` | N8 profiling spike + findings |
| `tests/equivalence/run_matrix.py` | Equivalence matrix (S3); CTest `equivalence_matrix`, label `slow` |
| `tools/batchsim/batchsim.cpp`, `docs/notes/batched_merging.md` | Independent batching simulator; the proof, measurements and risks (B0) |
| `report/` | Empty (contains only `.gitkeep`) |

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
