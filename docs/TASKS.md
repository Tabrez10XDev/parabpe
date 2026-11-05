# ParaBPE Task Board

Status: ⬜ todo · 🟨 in progress · ✅ done. Update this file whenever a task changes state.
Work through the tasks in **pick-up order**. A task can start once everything in its "Depends on" list is ✅.

| Pick-up order | Task | Status | Depends on |
|---|---|---|---|
| — | N1 Repo skeleton, CMake, CI, smoke tests | ✅ | — |
| — | N2 Spec + interfaces (draft) | ✅ (awaiting team review) | — |
| 1 | N4 Python reference + golden files | ✅ | — |
| 2 | N3 Data scripts | ✅ (TinyStories + WikiText run; enwik9 path written, not run) | — |
| 3 | S1 Chunker + serial pre-tokenizer | ✅ | N3, N4 |
| 4 | S2 WordTable + NaiveSerial + writer + `train()` | ✅ | S1 |
| 5 | S4 OMP pre-tokenizer + TSan CI | ✅ (TSan job written, runs once pushed) | S1 |
| 6 | S3 IncrSerial + equivalence harness | ✅ | S2 |
| 7 | N8 Profiling spike | ✅ | N3, N4 |
| 8 | N7 Bench harness | ✅ | — (needs the `timings.json` schema in SPEC §6 only) |
| 9 | G1 Profiling gate | ✅ (modest S6 + S10 chunking) | S3, S4 |
| 10 | S5 Parallel PairStats | ✅ | S3 |
| 11 | S7 NaivePar | ✅ | S2 |
| 12 | S6 IncrPar | ✅ (fork per large step, not a persistent region; see HANDOFF) | S5, G1 |
| 12b | S10 Parallel chunker | ✅ | S4, G1 |
| 13 | B0 Batched merging: plan, proof, `tools/batchsim` | ✅ | S6 |
| 14 | B1 Pure refactor into `src/incr_common.hpp` | ✅ | B0 |
| 15 | B2 `incr-batch` plumbing | ✅ | B1 |
| 16 | B3 Serial batched engine + verifier + tests | ✅ | B2 |
| 17 | B4 Hashed owners + parallel rounds (**mains power** for the kill check) | ⬜ | B3 |
| 18 | B5 IncrPar owner-map ablation | ⬜ | B4 |
| 19 | N6 Related-work notes | ⬜ | — |
| 20 | S8 Full benchmarks (on the Mac, **mains power**) | ⬜ | S4–S7, S10, N7, B5 |
| 21 | B6 Trace (`--trace`, P/E labels) | ⬜ | B3 |
| 22 | B7 ParaBPE Studio | ⬜ | B6, S8 |
| 23 | S9 Report | ⬜ | S8, B7 |
| — | N5 Get big-machine access (optional) | ⬜ **human task** | — |

---

## N4: Python reference (`reference/bpe_ref.py`) + golden files
**Goal:** a short, obviously-correct oracle that implements SPEC.md exactly. It is slow, and that is fine.

Files: `reference/bpe_ref.py`, `reference/make_golden.py`, `tests/golden/<case>/{input.txt,words.tsv,merges.txt,vocab.json}`.

Implementation:
- CLI: `bpe_ref.py --input F --vocab-size N --out DIR [--special "<|endoftext|>"]`. It writes
  `words.tsv`, `merges.txt` and `vocab.json` in the formats from SPEC §6.
- Split on special tokens (`re.escape`) and drop them. Run `regex.findall(GPT2_PAT, doc)` on each document.
  Count words as UTF-8 `bytes`.
- Merge loop: recount pairs from scratch every iteration (naive is fine here). Overlapping pairs are counted (§3).
- Selection: `max(pairs, key=lambda p: (count[p], (vocab[p[0]], vocab[p[1]])))`, where `vocab[id]` is a `bytes` object (§4).
- Apply each merge greedily, left to right, without overlaps. `new_id = 256 + n_special + i`.
- `bytes_to_unicode`: the standard GPT-2 table (copy it from the GPT-2 `encoder.py` logic).
- `make_golden.py` regenerates every case. Cases (each with vocab_size ≈ 256 + n_special + 30…100):
  `unicode` (accents, CJK), `emoji` (multi-byte sequences, ZWJ), `runs` (`aaaaaaa`, `abababab`),
  `ties` (many equal-count pairs), `multi_occurrence` (one pair appears several times in one word),
  `doc_edges` (whitespace at document start and end, empty docs, consecutive `<|endoftext|>`),
  `no_separator` (a single document).

Acceptance:
- Hand-checked merges for `runs`: on `aaaa` × 1, the first merge is `(a,a)` and the word becomes `[aa, aa]`.
- `vocab.json` + `merges.txt` load into HF `tokenizers` (`ByteLevelBPETokenizer(vocab, merges)`).
  Encoding a sentence and decoding it reproduces the text.
- Write a short `reference/README.md` stating the Python version and `regex` version used, because
  Unicode-version differences can matter.

---

## N3: Data scripts (`scripts/get_data.sh`, `scripts/slice_data.py`)
**Goal:** reproducible, normalized corpora + slices.

- `get_data.sh` downloads into `data/raw/<name>/`:
  - **TinyStories**: `TinyStoriesV2-GPT4-train.txt` from HF `roneneldan/TinyStories`. It is already separated by `<|endoftext|>`.
  - **WikiText-103**: raw v1 from HF (`Salesforce/wikitext`, `wikitext-103-raw-v1`).
  - **enwik9**: `http://mattmahoney.net/dc/enwik9.zip`.
- `slice_data.py --source {tinystories,wikitext,enwik9} --sizes 10M,100M,1G`:
  - Decode bytes with `errors="replace"`, so the output is valid UTF-8 (SPEC §1).
  - Insert `<|endoftext|>` between documents:
    - WikiText: a new article starts at a line matching `^ = [^=].* = $`.
    - enwik9: split per `<page>`.
  - Write `data/<source>.full.txt` and `data/<source>.<size>.txt`. Slices take **whole documents
    only**, cut at a document boundary at or below the size limit.
  - Write `data/CHECKSUMS.sha256`.
- Acceptance: every output passes `bytes.decode("utf-8")` (strict). Rerunning produces identical checksums.
- Done 2026-11-01. Run with `.venv/bin/python -I scripts/slice_data.py --source tinystories --sizes 10M,100M,1G --no-full`
  (`--no-full` avoids a 2.2 GB copy) and `--source wikitext --sizes 10M,100M` (full = 540 MB). Sizes are decimal (M = 10^6).

---

## S1: Chunker + serial pre-tokenizer
Files: `src/corpus.cpp` (`make_work_units`), `src/pretok_serial.cpp`, a new `include/parabpe/regex.hpp` +
`src/regex.cpp` (RAII wrappers), `src/main.cpp` (`--dump-words` flag), `tests/unit/test_chunker.cpp`,
`tests/unit/test_pretok.cpp`.

Implementation:
- `make_work_units`: scan for special tokens. With several specials, take the leftmost match at
  each position, preferring the longest. Documents are the spans between separators; **skip empty documents**.
  Pack documents into units in order: close a unit once its bytes ≥ `target_bytes`. A document larger than
  the target becomes its own unit. The result must not depend on the thread count.
- Validate UTF-8 once over the whole corpus with a small validator in `corpus.cpp`, and throw on invalid input.
- `regex.hpp`:
  - `class Regex`: owns the `pcre2_code`, JIT-compiled. It is immutable and safe to share across threads.
  - `class Matcher`: per thread; owns the `pcre2_match_data` (+ a JIT stack if needed).
  - `void for_each_match(std::string_view, F)` uses `PCRE2_NO_UTF_CHECK`.
- `SerialPreTokenizer`: count into `ankerl::unordered_dense::map<std::string_view, uint64_t>`. The views
  point into the mmap. Convert to `WordCounts` **sorted by bytes** (SPEC §6).
- Add `train --dump-words` → writes `<out>/words.tsv`.

Also (from the independent plan review, 2026-10-31):
- `for_each_match` **throws** on any negative PCRE2 return code and on any gap (`ov[0] != pos`). Give each `Matcher`
  a larger JIT stack, and test with ~1 MB runs of whitespace and of letters.
- Add a repeatable `--special` flag to `train` (the default `<|endoftext|>` applies only when none is given), so the
  `multi_special` golden case can run. Each case's specials are in `tests/golden/<case>/case.txt`.
- `write_words` is declared in `io.hpp`. `--dump-words` writes `words.tsv` and exits **before** `build_word_table`.
- `tests/unit/test_unicode_parity.cpp`: check PCRE2's `\s`, `\p{L}` and `\p{N}` against
  `tests/golden/unicode_classes.txt` for every code point (surrogates skipped).
- Compare bytes through `std::string_view`/`memcmp`, never through signed `char` (this also matters for S2/S3 tie-breaks).

Acceptance: unit tests for the chunker edge cases (no separators, leading/trailing/consecutive separators,
a document larger than the target, several special tokens). `words.tsv` must be byte-identical to `bpe_ref.py`
(`--words-only` on big inputs) for every golden case, the 10 MB TinyStories slice and one non-ASCII 10 MB slice
(WikiText or enwik9).

Done 2026-11-02. All acceptance checks pass (see HANDOFF). Found on the way: PCRE2's `\s` (UCP) also matches U+180E,
which is not whitespace in Unicode 16, so `Regex` compiles `\s`/`\S` as `\p{White_Space}`/`\P{White_Space}`
(`unicode_whitespace_pattern` in `regex.hpp`). After that, PCRE2 matches Python on every code point.

---

## S2: WordTable + NaiveSerial + writer + `train()` library function
Files: `src/word_table.cpp`, `src/merge_naive.cpp` (serial path), `src/io.cpp`, a new
`include/parabpe/trainer.hpp` + `src/trainer.cpp`, `src/main.cpp`, `tests/unit/test_golden.cpp`,
`tests/interop/check_hf.py`.

Implementation:
- Move the pipeline out of `main.cpp` into `TrainResult train(const std::string& input, const TrainConfig&)`.
  `main` then only parses arguments and writes files, and the golden tests call `train()` directly.
- **Interface change (allowed):** `MergeEngine::run(WordTable&, const TrainConfig&, PhaseTimings&)`
  so engines can report `pairs` and `merge` separately.
- Engines keep their own `std::vector<std::string> token_bytes` for tie-breaks (SPEC §4).
  Number of merges = `vocab_size − 256 − n_special`. Stop early if no pairs remain.
- Merge application + count updates exactly as in SPEC §3.
- `io.cpp`: the `bytes_to_unicode` table, `build_vocab`, and the writers in SPEC §6 formats. Write JSON by hand
  (escape `"`, `\` and control characters) to avoid a new dependency.
  - In `write_timings`, get the hardware info from `sysctl` on macOS and `/proc/cpuinfo` + `lscpu` on Linux.
  - Get the git SHA from `git rev-parse HEAD`; use `"unknown"` if that fails.
- `test_golden.cpp`: for each `tests/golden/*` case, read `case.txt`, call `train()` with NaiveSerial and compare
  `merges.txt` **and `vocab.json`** byte for byte. Pass the golden dir as a compile definition (`PARABPE_GOLDEN_DIR`).
  The exact `vocab.json` bytes (compact, no trailing newline, escape rules) are in `reference/README.md`.
- `check_hf.py`: load our output into HF and compare HF token ids against `bpe_ref` encodings of held-out text.
  (Add a C++ encoder later only if needed.)

Acceptance: all golden tests pass, including under ASan, and `check_hf.py` passes.

Done 2026-11-02 (ASan on CI only; it can't run on this Mac, so UBSan was used locally). As built:
- `include/parabpe/trainer.hpp`: `count_words` (phase 1), `train_text` (in-memory), `train` (file).
- `io.hpp` returns strings: `format_words/merges/vocab/timings` + `write_file`; `format_vocab(vocab, n_special)`.
  `write_words` is gone (S1 callers now use `format_words`).
- `src/bpe_common.hpp` (internal, for every engine): `num_merges`, `initial_token_bytes`, `better_pair` (SPEC §4),
  `apply_merge` (SPEC §3). **S3/S6/S7 must use these**, so tie-breaks and merge rules can't drift apart.
- `tests/unit/test_golden.cpp` loops over `kEngines`: add each new engine there.
- `timings.json` adds `performance_cores` / `efficiency_cores` to `hardware` on Apple Silicon (extra keys only).

---

## S4: OMP pre-tokenizer + TSan CI
Files: `src/pretok_omp.cpp`, `.github/workflows/ci.yml`, `tests/unit/test_pretok.cpp`.

Implementation:
- `#pragma omp parallel`: each thread has its own `Matcher` and T local maps (**pre-bucketed** by the **low** bits of the
  hash: `bucket = hash & (B − 1)`, B = T rounded up to a power of 2). Not the high bits: `ankerl` maps pick their slot
  from the high bits, so high-bit bucketing made the local maps degenerate (50× slower at 3 threads, found in S4). Use `#pragma omp for schedule(dynamic,1)` over work units.
- After the implicit barrier, thread t merges bucket t from every thread's local maps into its own shard map.
  No locks are needed. Concatenate the shards and sort by bytes.
- Add a variant flag or config, used for the report, for a `critical`-section merge to compare against.
- CI: add a `tsan` job on ubuntu with `clang` + `libomp-dev` (LLVM's runtime), built with `-DPARABPE_TSAN=ON`, and
  run the tests with `OMP_NUM_THREADS=4`.

Reuse from S1: `Regex` (shared) + one `Matcher` per thread (`parabpe/regex.hpp`), `detail::sorted_word_counts`
(`src/strategies.hpp`), and the golden `words.tsv` test in `tests/unit/test_pretok.cpp` (extend it to `PretokStrategy::OMP`).

Acceptance: `words.tsv` is identical to the serial output for threads {1,2,3,4,8} on all golden cases and the 10 MB slice. TSan is clean.

---

## S3: IncrSerial + equivalence harness
Files: `src/merge_incr.cpp` (serial path), `tests/equivalence/run_matrix.py`, a CTest entry labelled `slow`.

Implementation:
- `counts: map<PairKey,int64>`, `index: map<PairKey, vector<uint32_t>>` (word ids, duplicates allowed),
  a lazy max-heap of `(count, pair)` with the SPEC §4 comparator.
- For each merge:
  - Pop entries until the top is live (its count equals `counts[pair]` and is > 0). Stop if the heap is empty.
  - For each word id in `index[pair]`: skip it if `last_touched == merge#`, otherwise stamp it. Skip it if the word
    no longer contains the pair. Otherwise apply the whole-word diff and record the deltas.
  - Apply the deltas. Push changed pairs with positive counts to the heap. Append the word id to `index[p]` for
    every new pair. Erase `index[pair]`.
  - Rebuild the heap from `counts` when its size exceeds 4× the number of live pairs.
- `run_matrix.py --input data/tinystories.10M.txt --vocab-size 4000`: runs every implemented
  (pretok, merge) combination × threads {1,2,3,4,8,16} and compares the sha256 of `merges.txt`.
  NaiveSerial is the oracle. The exit code is non-zero on any mismatch.

Acceptance: golden tests pass with IncrSerial. The matrix is identical on 10 MB. IncrSerial is ≥10× faster than NaiveSerial at vocab 4k.

Reuse from S2: `src/bpe_common.hpp` (`better_pair`, `apply_merge`, …), `train_text`, and `kEngines` in `test_golden.cpp`.
Reference numbers: NaiveSerial on `tinystories.10M` at vocab 1000 = 0.27 s merge phase (`runs/s2/ts10M_v1000`).

---

## N8: Profiling spike (`scripts/profile_phases.py`)
- Time `bpe_ref.py`'s phases (pre-tokenize, count, merges at vocab 1000) on the 10 MB slice.
- Time HF `tokenizers` BPE training at `RAYON_NUM_THREADS=1` and at N threads on 100 MB.
- Write the results to `docs/notes/profiling.md` (a table + 3 bullet takeaways).

## N7: Bench harness (`bench/run_bench.py`, `bench/plot.py`, `bench/configs/*.yaml`)
- YAML grid: `inputs`, `threads`, `pretok`, `merge`, `vocab_size`, `reps` (default 5). The first run of each
  configuration is a discarded warm-up.
- Set `OMP_PROC_BIND=close OMP_PLACES=cores` (ignored on macOS, effective on Linux). Use `numactl` if it is present
  and the machine has more than one NUMA node.
- Default thread grid on the Mac: {1,2,3,4,5,6,8,11}. Record whether the run was on mains power.
- Append each run's `timings.json` (SPEC §6) as a row to `results/<config>.csv`.
- `plot.py`: per-phase speedup vs IncrSerial, a stacked phase-time bar, an Amdahl fit, and weak scaling for phase 1.
  Use median lines with min/max bands.
- Develop against hand-written fake `timings.json` files until S2 exists.

## G1: Profiling gate
On 100 MB and 1 GB, measure the C++ IncrSerial phase breakdown and the OMP pre-tokenizer speedup. Record the results in
`docs/notes/profiling.md`. Decide with the user how much effort S6 gets.

## S5: Parallel PairStats
Partition the words statically by owner thread. Each thread counts pairs and builds a **private** inverted index for
the words it owns. Reduce the global counts by bucket, then build the heap. Acceptance: equivalence matrix passes.

## S7: NaivePar
NaiveSerial with `omp for` over words, thread-local pair maps and a bucketed reduction. Use small configurations only
(10 MB, vocab ≤ 4k). Acceptance: equivalence matrix passes.

## S6: IncrPar
- One persistent `omp parallel` region for the whole merge loop:
  1. The master selects the pair, then a barrier.
  2. Each owner applies the merge to its affected words through its private index and writes deltas to its
     64-byte-aligned buffer, then a barrier.
  3. Bucketed delta reduce; the master pushes the changed pairs to the heap.
- Threshold on **tokens touched**: below it, the master does the step alone.
- Acceptance: equivalence matrix passes and TSan is clean. Expect a 1–2× speedup on phase 3 and report it honestly.

## S10: Parallel chunker (added at G1)
Files: `src/corpus.cpp`, `tests/unit/test_chunker.cpp`.
- `validate_utf8` in parallel: split the text into T byte ranges, move each start forward past continuation bytes
  (`10xxxxxx`), and validate the ranges independently. Report the **lowest** failing offset, so the error message
  does not depend on T.
- Special-token split in parallel: each thread finds the occurrences in its range plus an overlap of
  (longest special − 1) bytes, keeping only matches that start in its own range. Then concatenate in order and apply
  the leftmost/longest rule serially over the merged candidate list. Work units must come out **identical** to the
  serial chunker for every T, because they decide nothing about the result but are compared in tests.
- `make_work_units` takes the thread count (`cfg.threads`); the serial path stays the default for `--pretok serial`.

Acceptance: work units identical to serial for threads {1,2,3,4,8,16} on all golden inputs, on random texts with
specials straddling range borders, and on the 10 MB slices. Invalid UTF-8 reports the same offset for every T.
The equivalence matrix passes. Phase `chunk` speedup is measured in S8.

## Upgrade: batched merging
The idea, the proof and the measurements are in `docs/notes/batched_merging.md`. Every exactness change gets a mutation
check. After each task: tests green in `build` and `build-debug`, docs updated, commit, push. IncrSerial stays the
baseline; its phase 3 on `wikitext.full` v32k must stay about 0.66–0.68 s.

### B0: Record the plan
`docs/notes/batched_merging.md`, these cards, the push rule in `CLAUDE.md`, `tools/batchsim/batchsim.cpp`, and the CI
TSan option.
Done 2026-11-03. `batchsim` reproduces all six relaxed-rule round counts; the closed form needs 2,561 rounds on
`wikitext.full` v32k. See HANDOFF.

### B1: Pure refactor
Move `ShardedCounts`, `Owner`, `LocalCounts`, `partition_words`, `count_serial`, `count_parallel`, `merge_owner`,
`MergeStep`, `tokens_touched`, `ThreadDelta`, `ShardDelta` (and `HeapEntry`, `contains_pair`, `index_word`, which they
need) into `src/incr_common.hpp`. `parallel_step` becomes a free template:
`template <class MergeFn> void parallel_step(WordTable&, std::vector<Owner>&, ShardedCounts&, MergeFn&&, std::vector<ThreadDelta>&, std::vector<ShardDelta>&, int threads)`.

Acceptance: behaviour identical; tests and matrix identical; IncrSerial phase 3 within noise, old vs new binary.

### B2: Plumbing
`MergeStrategy::IncrBatch` (CLI `incr-batch`) through `types.hpp`, `main.cpp`, `io.cpp`, `strategies.{hpp,cpp}` and
`test_smoke.cpp`. `TrainConfig`: `verify_batches=false`, `owners_per_thread=4`, `batch_scan_cap=256`, `trace_path=""`.
`MergeStats` in `TrainResult`: rounds, max_batch, parallel_rounds, histogram {1,2,3-4,5-8,9-16,17-32,33-64,65-128,129+},
stop reasons {disjoint, tie, self, budget, capped}, select_s, order_s, changed_pairs. `MergeEngine::stats()`;
`format_timings` writes `"batch": {...}` only for incr-batch. Flags `--verify-batches`, `--owners-per-thread`,
`--batch-scan-cap`, `--trace`. Stub: `std::logic_error("IncrBatch: not implemented (B3)")`.

### B3: Serial batched engine + verifier + tests
Files: `src/merge_batch.cpp`, `src/batch_select.hpp`, `tests/unit/test_batch.cpp`.
- `select_round` on the lazy heap; pure `choose_batch(prefix, tail, complete, tok_mark, stamp)` with strict, relaxed and
  closed-form rules behind an internal enum, verified in that order. **j is 0-based.**
- `apply_batch`: one pass per word via `first_slot[tok]` and `second[slot]`; one whole-word diff per word per round.
- `--verify-batches`: a shadow sequential oracle on a copy of the table.
- Tests: `choose_batch` cases (tie with p1, tie only with pK, self-pair, cap 0 and 1, budget 1, the prefix-stopping pair
  in T); `apply_batch` vs sequential, with a self-pair in the middle; adversarial corpora, including `eax`×2 + `df`×2.
- Add to `kEngines`/`is_parallel`, the `Engines` suite, the random list with `verify_batches=true`, and `run_matrix.py`.
- Mutations that must fail: always k=K; self-pair mid-batch; `<=` in g; bytes push after heap push; 1-based j; the
  stopping pair left out of T.
- Settle the `batch_scan_cap` default by measuring: at 256, `wikitext.full` v100k takes 35,237 rounds against 21,295
  uncapped. **Settled: 4096.**

Acceptance: identical everywhere; `--verify-batches` passes on both 10 MB slices and `wikitext.full` v32k; rounds ≤ 3,409
there (the closed form predicts 2,561), cross-checked with batchsim; changed pairs and select/order time logged.

### B4: Hashed owners + parallel rounds
`partition_words_hashed`: owner = mix64(w >> 4) % n, increasing `words` list, n = owners_per_thread·T (1 at T=1),
`schedule(dynamic,1)`. A round goes parallel when Σ tokens_touched ≥ `par_threshold_tokens`. Zero-count erasure moves to
a periodic parallel sweep. Threshold sweep {1k, 4k, 16k, 64k}.

**Kill check (mains power, median of 3):** phase 3 at 5 threads on `wikitext.full` < 1.3× IncrSerial at **both** vocab
32k and 100k stops the optimising. Then ship the theorem, the span reduction and the imbalance fix, and move to B6, B7, S9.

Acceptance: matrix includes `incr-batch` with 0 mismatches; `--verify-batches` passes at 5 and 11 threads at v32k and v100k.

### B5: IncrPar owner-map ablation
`--owner-map contiguous|hashed` for IncrPar (default contiguous). Log busiest-owner/mean tokens on steps ≥ 4096 in
`MergeStats`, to measure the 2.30 → ~1.07 claim directly.

### N6: Related work
Prior multi-merge and parallel BPE training: HF tokenizers, SentencePiece, YouTokenToMe, minbpe, papers. Write
`docs/notes/related_work.md` and decide what the novelty claim may say.

### S8: Full benchmarks (needs mains power)
`bench/configs/batch.yaml`: incr-serial, incr-par, incr-batch × omp × threads {1,2,3,4,5,6,8,11} × {wikitext.full,
tinystories.1G} × vocab {32000, 100000}. `bench/configs/scale.yaml`: full TinyStories (2.2 GB) and 3.2 GB WikiText. Also
main, naive, reduce, threshold and weak, all with `--require-ac`, plus the HF comparison again on mains power. Extend
`run_bench.py` (`PARALLEL`, the threshold rule in `grid()`, `rounds`/`max_batch`) and `plot.py` (`STRATEGY_COLORS`, span
chart, batch-size histogram). Commit `results/`. See also `docs/PLAN.md` §6.

### B6: Trace
`--trace FILE` writes Chrome Trace Event JSON from per-thread `alignas(64)` holders: zero cost off, under 2% on. Kinds:
CHUNK, PRETOK_UNIT, SELECT, SERIAL_APPLY, OWNER_MERGE, SHARD_REDUCE, ZERO_SWEEP, ORDER_UPDATE. P/E labels from
`pthread_cpu_number_np` and `ioreg` (ids 0–5 = E, 6–10 = P on this Mac): a sample, not pinning. This lands after S8, so
re-measure IncrSerial and `incr-batch` afterwards, and rerun S8 configs if the off cost isn't within noise.

### B7: ParaBPE Studio
`studio/index.html` (vanilla JS, no build) + `studio/serve.py` (stdlib, bound to 127.0.0.1, inputs under `data/` only).
Replay with swimlanes and P/E badges, merges landing per round, a merges/rounds counter and span curve, an exactness badge
(sha256 against IncrSerial), and speedup charts from `results/*.csv`.

### S9: Report
Title: *ParaBPE: Provably Exact Parallel BPE Training — Batched Merges That Break the Sequential Chain.* `report/report.tex`;
every number traced to `results/`. See also `docs/PLAN.md` §9.

## N5 (human)
- **N5 (optional):** only if time and budget allow, rent a machine with ≥16 *physical* cores for an extra scaling
  point (e.g. AWS `c7a.4xlarge` / `c7g.4xlarge`, which have no SMT). The main results come from the Mac.
- **N6:** see the card above; also cover Sennrich 2016, GPT-2 and the CS336 BPE notes.
