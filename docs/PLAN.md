# ParaBPE: Architecture, Flow & Execution Plan


## Context
This is a Parallel Computing course project. **ParaBPE** trains a byte-level BPE tokenizer (the GPT-2/Llama style) in parallel on multicore CPUs. Success criteria:
1. **Exactness**: every parallel variant produces exactly the same merges as the serial version, compared byte for byte, at every thread count.
2. **Honest speedup**: measured against the *best* serial implementation (IncrSerial), with strong scaling, weak scaling for phase 1, and an Amdahl analysis.
3. **A clear story in the report**: phase 1 (pre-tokenization) and phase 2 (pair statistics) scale well. Phase 3 (the merge loop) is bound by Amdahl's law, and we measure and explain why.

> v2 incorporates an independent principal-level review. The main fixes: a chunking rule that keeps output exact, a precise pair-update rule, an honest MVP baseline, a redesigned IncrPar, workable TSan, a feasible benchmark grid, and a cut scope.

---

## 1. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Core | **C++20** | |
| Parallelism | **OpenMP** | `std::thread` and MPI are optional, only if time is left |
| Regex | **PCRE2** with JIT | Gives each thread its own `match_data` and JIT stack; validates UTF-8 once, then matches with `PCRE2_NO_UTF_CHECK` |
| Hash maps | `ankerl::unordered_dense` (header-only) | |
| Build / tests | CMake ≥3.20, GoogleTest (FetchContent), CTest, CLI11 | |
| Reference / bench / plots | Python 3.11: `regex`, pandas, matplotlib, `tokenizers`, `tiktoken` | |
| CI | GitHub Actions: GCC build + tests; **clang + LLVM libomp** TSan job | GCC's OpenMP runtime gives false race reports under ThreadSanitizer |
| Profiling | `perf` + flame graphs, ASan | |
| Report | LaTeX / Overleaf | |

macOS development: `brew install libomp pcre2 cmake llvm`. Final numbers come from the **developer Mac (M3 Pro, 5 P + 6 E cores)**; see `docs/HANDOFF.md` for its measured scaling. A ≥16-physical-core Linux machine is an optional extra. Cloud "16 vCPU" usually means only 8 physical cores, so check before booking (AWS `c7a`/`c7g` have no SMT).

---

## 2. Spec (frozen in week 1; exactness depends on it)

### 2.1 Input normalization (done by the data-prep script, once)
- Corpus = documents separated by the special token `<|endoftext|>`. For corpora without separators (WikiText, enwik9), `slice_data.py` inserts one between articles.
- Invalid UTF-8 is replaced with U+FFFD during preparation, so every engine and the Python reference see valid UTF-8. The C++ code validates once and then matches with `NO_UTF_CHECK`.

### 2.2 Special tokens & chunking
- Split the corpus on special tokens and **drop them before the regex runs**. They get fixed ids after the 256 byte tokens and never take part in merges.
- **The unit of work is a document.** Documents are grouped into ~1 MB work units by a deterministic rule that ignores the thread count. Threads pull work units dynamically. The regex never sees text across a document boundary, in the serial code, the parallel code or Python alike. Splitting on `\n\n` is not allowed, because `\s+(?!\S)` would split differently.

### 2.3 Pair counting & merge application
- **Counting:** count every adjacent pair, including overlapping ones (`aaa` holds two `(a,a)` pairs), weighted by the word's frequency.
- **Applying a merge `(x,y)→z` to a word:** greedy, left to right, without overlaps. `aaaa` becomes `zz`, and `aaa` becomes `z a`.
- **Count update for each affected word (diff the whole word):** subtract every old pair of the word × freq, rewrite the word, then add every new pair × freq. No patching of neighbours one at a time; that is where the bugs live.

### 2.4 Tie-break
Best pair = highest count. On a tie, take the lexicographically largest **tuple** `(bytes(a), bytes(b))`, compared as a tuple and not as joined bytes. The heap comparator, the parallel max-reduction and `bpe_ref.py` all implement this exact rule.

### 2.5 Output format
- `merges.txt` and `vocab.json` use GPT-2's `bytes_to_unicode` mapping, so spaces and newlines in tokens survive the file format.
- These files can be loaded directly by HF `tokenizers` (ByteLevel BPE).

---

## 3. Architecture

### 3.1 Pipeline
```
 prepared corpus (valid UTF-8, <|endoftext|> separated)
     │ mmap
     ▼
 Chunker ── split on special tokens → docs → fixed ~1MB work units (independent of T)
     ▼
 PHASE 1  PreTokenizer (OMP, dynamic over work units)
          per thread: PCRE2 split → local map<bytes,count>, PRE-BUCKETED into T buckets
                      by high hash bits
          barrier → thread t merges bucket t from all T local maps (no locks, O(N) total work)
     ▼
 WordTable (SoA: flat tokens[], offset[], len[], freq[], last_touched[])
          words are statically partitioned across threads (owner computes)
     ▼
 PHASE 2  PairStats (OMP): each thread counts pairs + builds a PRIVATE inverted index
          for its own words → global counts via a bucketed reduction → build the max-heap
     ▼
 PHASE 3  MergeEngine: sequential outer loop; parallel inner work only when worthwhile
     ▼
 Writer (merges.txt, vocab.json, timings.json) + Encoder (used for validation)
```

### 3.2 Core interfaces
```cpp
using TokenId = uint32_t;  using PairKey = uint64_t;   // (a<<32)|b
struct Merge { TokenId a, b, new_id; };
struct WordCounts { std::vector<std::string> words; std::vector<uint64_t> freq; };
struct WordTable  { std::vector<TokenId> tokens; std::vector<uint32_t> offset, len, last_touched;
                    std::vector<uint64_t> freq; };
struct TrainConfig { size_t vocab_size; int threads; std::string pattern;
                     std::vector<std::string> special_tokens;
                     PretokStrategy pretok;  MergeStrategy merge;   // see §3.3
                     size_t par_threshold_tokens; };
struct PhaseTimings { double chunk, pretok, build, pairs, merge, total; };
class PreTokenizer { public: virtual WordCounts run(const Corpus&, const TrainConfig&) = 0; };
class MergeEngine  { public: virtual std::vector<Merge> run(WordTable&, const TrainConfig&) = 0; };
```

### 3.3 Merge engine variants

| Variant | What it does | Role | Run on |
|---|---|---|---|
| **NaiveSerial** | Recounts every pair for every merge | Oracle on medium-size inputs | 10 MB, vocab ≤ 4k only |
| **IncrSerial** | Inverted index + whole-word diff updates + lazy max-heap | **Official baseline** | All sizes |
| **NaivePar** | NaiveSerial with an OMP `for` + reduction | Shows that "it scales but is limited by memory bandwidth and loses to IncrSerial" | 10 MB, vocab ≤ 4k only |
| **IncrPar** | Described below | The main parallel attempt at phase 3 | All sizes |

**IncrSerial details**
- **Lazy heap:** pop an entry, check it against the live count, and discard it if stale. Rebuild the heap when it holds more than 4× the live pairs. Stop early when no pairs are left.
- **Inverted index:** `index[pair]` is a list of word ids. Use a `last_touched[word] == merge#` stamp to skip duplicates. Free `index[merged_pair]` after each merge.

**IncrPar (redesigned)**
- **One persistent `omp parallel` region** for the whole merge loop. Within each merge, the steps are:
  1. **Select.** The master thread pops the best pair from the heap; the other threads wait at a barrier.
  2. **Apply.** Each thread processes the affected words *it owns*, using its private inverted index (owner computes). Each thread writes count changes to its own delta buffer, padded to 64 bytes to avoid false sharing.
  3. **Reduce.** Deltas are combined by bucket into global counts. The master pushes the changed pairs to the heap.
- **Adaptive threshold on tokens touched** (not words affected). Below the threshold, the master does the whole step alone and the other threads skip it. The threshold sweep goes in the report.
- **Expectation, stated up front:** phase 3 IncrPar speedup is about 1–2×. The selection step is inherently serial, and later merges touch very few tokens. The end-to-end speedup comes from phases 1 and 2. This Amdahl breakdown is the centrepiece of the report.

### 3.4 CLI
```
parabpe train --input data/wt103.txt --vocab-size 32000 --threads 16 \
              --pretok omp --merge incr-par --par-threshold-tokens 4096 --out runs/x/
parabpe encode --model runs/x --input heldout.txt --out ids.bin
```
`timings.json` records the phase times, CPU model, physical and logical core counts, SMT on or off, the git SHA and the full config.

---

## 4. Repo Layout
```
parabpe/
├── CMakeLists.txt                # options: PARABPE_TSAN (clang+libomp), PARABPE_ASAN
├── include/parabpe/              # types, corpus, pretokenizer, merge_engine, io
├── src/  corpus.cpp  pretok_serial.cpp  pretok_omp.cpp  word_table.cpp  pair_stats.cpp
│         merge_naive.cpp  merge_incr.cpp  io.cpp  encoder.cpp  main.cpp
├── reference/bpe_ref.py          # same spec (§2), tiny-corpus oracle
├── tests/  unit/  equivalence/  golden/
├── bench/  run_bench.py  plot.py  configs/*.yaml
├── scripts/ get_data.sh  slice_data.py  profile_phases.py
├── .github/workflows/ci.yml
└── report/
```

---

## 5. Verification
1. **Golden tests (tiny corpora, Python oracle):** Unicode, emoji, long runs (`aaaaaaa`), words with many ties, multiple occurrences of a pair in one word, whitespace runs at document edges.
2. **Equivalence matrix (10 MB, C++ NaiveSerial as oracle):** every strategy × threads {1,2,3,4,8,16}. All `merges.txt` hashes must be identical.
3. **Interop check:** load our `merges.txt` and `vocab.json` into HF `tokenizers`. Encode held-out text with HF and with our encoder; the token ids must be identical. (A decode/encode round trip alone always passes for byte-level BPE, so it proves nothing.)
4. **Sanitizers:** ASan on everything. TSan with clang + LLVM libomp in CI.
5. **External sanity check:** HF trained with the same pattern gives more than 95% vocabulary overlap and a compression ratio within ~1%. Exact equality is not expected, because HF breaks ties differently.

---

## 6. Benchmark Methodology
- **Main grid:** IncrSerial and IncrPar × pretok {serial, omp} × threads {1,2,3,4,5,6,8,11} on the Mac (P-core region 1–5 reported separately from 6–11; up to the physical core count on any extra machine) × {100 MB, 1 GB}, vocab 32k. Naive variants run only on 10 MB with vocab ≤ 4k.
- **Runs:** 5 per configuration, report the median with a min/max band. Use a **warm page cache** (do one discarded run first) and say so in the report.
- **Controls:** `OMP_PROC_BIND=close OMP_PLACES=cores` (no effect on macOS, which has no thread pinning; say so in the report); on the Mac, run on mains power with nothing else running; SMT off or ignored; `numactl --cpunodebind=0 --membind=0` on machines with two CPU sockets. Record turbo state, since it usually can't be disabled in the cloud.
- **Plots:**
  - Speedup per phase relative to IncrSerial
  - Stacked bars of time per phase
  - Amdahl fit (estimated serial fraction)
  - Weak scaling for **phase 1 only**
  - Sweep of the token threshold
  - OMP schedule comparison
  - Bucketed merge vs. `critical` merge
  - Bandwidth ceiling for NaivePar
  - Comparison with HF `tokenizers`

---

## 7. What Can Start RIGHT NOW (parallel, no dependencies)

| # | Task | Output | Effort |
|---|---|---|---|
| **N2** | **Freeze the spec (§2) and interfaces (§3.2), the whole team.** This is the only coordination point. | `SPEC.md` + `include/parabpe/*.hpp` stubs | ½ day |
| **N1** | Repo skeleton: CMake (OpenMP, PCRE2, GoogleTest, CLI11, unordered_dense) + an `omp parallel` smoke test + CI with GCC and clang-TSan jobs | Green CI | ½–1 day |
| **N3** | `get_data.sh` + `slice_data.py`: TinyStories, WikiText-103, enwik9 → UTF-8 clean-up, `<|endoftext|>` insertion, 10 MB / 100 MB / 1 GB slices with checksums | Reproducible `data/` | ½–1 day |
| **N4** | `reference/bpe_ref.py` implementing §2 exactly + tiny golden corpora + expected `merges.txt` | `tests/golden/` | 1 day |
| **N5** | *(Optional)* Rent a big machine for an extra scaling point: an instance with at least 16 *physical* cores | SSH access | Only if time and budget allow |
| **N8** | **Profiling spike:** `profile_phases.py` times the regex, counting and merge phases of the Python reference on 10 MB, and HF `tokenizers` at 1 and N threads on 100 MB | Rough share of time per phase → confirms where effort should go | ½ day |
| **N6** | Related-work notes: Sennrich 2016, GPT-2, HF `tokenizers`, SentencePiece, minbpe, CS336 BPE notes | Bibliography | Any time |
| **N7** | `run_bench.py` + `plot.py` built against the CLI contract with a fake `timings.json` | Harness ready | 1 day |

Suggested split for 3 people:
- **A:** N1, then S1
- **B:** N4 + N8, then S2/S3
- **C:** N3 + N5 + N7, then S4

N2 is done together on day 1.

---

## 8. Sequenced Build Steps

| Step | Work | Depends on | Exit criterion |
|---|---|---|---|
| S1 | mmap + chunker (§2.2) + serial pre-tokenizer (PCRE2) | N1, N2, N3 | Word counts equal `bpe_ref.py` on golden and 10 MB data |
| S2 | WordTable + serial PairStats + **NaiveSerial** + writer (`bytes_to_unicode`) | S1, N4 | Golden tests pass; files load in HF |
| S3 | **IncrSerial** (index, whole-word diff, lazy heap, stamps) | S2 | Same merges as NaiveSerial on 10 MB |
| **G1** | **Profiling gate:** C++ phase breakdown on 100 MB and 1 GB | S3, S4 | Decide how much effort IncrPar gets |
| S4 | **OMP pre-tokenizer** + bucketed shard merge | S1 | Equivalence matrix passes; first speedup plot |
| S5 | **Parallel PairStats** (owner partitions, private indexes) | S3 | Equivalence passes |
| S6 | **IncrPar** (persistent region, token threshold) | S5, G1 | Equivalence passes, TSan clean |
| S7 | NaivePar (small configurations) | S2 | Equivalence passes |
| S8 | Full benchmark on the Mac (+ optional big machine) + plots | S4–S7, N7 | All plots from §6 |
| S9 | Report + slides | S8 | Submitted |
| opt | `std::thread` variant / MPI phase 1 | only if ahead of schedule | — |

S4 runs alongside S2 and S3. S7 is small and fills gaps.

---

## 9. Timeline (2–3 people, ~8–10 hours a week each)

| Week | Work |
|---|---|
| 1 | N1–N8 (all of the "right now" tasks) |
| 2 | S1, S2 ‖ start S4 |
| 3 | S3 ‖ finish S4 → **G1 gate** |
| 4 | S5, S7 |
| 5 | S6 (the riskiest step) |
| 6 | S8 benchmarks + tuning |
| 7 | S9 report and slides + buffer |

**Total: about 7 weeks (120–160 person-hours).**
- **MVP (about 4.5 weeks):** S1, S2, **S3**, S4, S5, S8 (reduced), S9. This still gives an honest speedup against IncrSerial, with the speedup coming from phases 1 and 2.
- **Cut from scope:** the `std::thread` variant, MPI, and naive variants at large sizes.
- **Main risks:**
  - S6 races and mismatches → mitigated by the equivalence matrix and TSan from day 1.
  - Exactness breaking at document edges → mitigated by golden cases and a frozen spec.
  - Heterogeneous P/E cores and no pinning on the Mac → report the P-core region separately and use dynamic scheduling where possible.
