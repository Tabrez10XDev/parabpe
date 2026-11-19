# Handoff

What doesn't change often: the machine, the decisions, the lessons, and a one-line log per task (task status is in
[TASKS.md](TASKS.md)). The full per-task notes are in git history
(`git show 6b378a8:docs/HANDOFF.md`).

## Environment (developer Mac)
- Apple M3 Pro, 5 performance + 6 efficiency cores, 18 GB, macOS 27. Compiler: Apple clang 17 (`/usr/bin/c++`); the
  `clang` on PATH is the swiftly toolchain, which CMake doesn't use. `cmake` from MacPorts, `libomp` from Homebrew
  (`CMakeLists.txt` asks `brew --prefix libomp`; keep that block).
- **PCRE2 must be 10.45** (SPEC pins Unicode 16). Homebrew's is 10.49 (Unicode 17) and fails
  `PCRE2.UnicodeVersionIsPinned`. Build it once with `scripts/build_pcre2.sh ~/pcre2` (as CI does) and configure with
  `-DPCRE2_INCLUDE_DIR=$HOME/pcre2/include -DPCRE2_LIBRARY=$HOME/pcre2/lib/libpcre2-8.a`. CMake warns on a newer PCRE2.
- **Sanitizers:** ASan and TSan runtimes hang or fail at startup on this macOS; both run in CI (Ubuntu). Locally use
  UBSan: `-DCMAKE_BUILD_TYPE=Debug -DPARABPE_NATIVE=OFF` + the PCRE2 flags +
  `-DCMAKE_CXX_FLAGS="-fsanitize=undefined -fno-sanitize-recover=undefined" -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=undefined`.
  The Debug build also turns on libc++/libstdc++ bounds checks.
- **Python:** always `.venv/bin/python -I` (the system Python lacks `regex`; other `regex` versions break the pin).
- **TeX:** `tectonic` (Homebrew): `cd report && tectonic report.tex`.
- First configure downloads GoogleTest, CLI11 and unordered_dense (FetchContent): needs network.

## Benchmark machine and hygiene
- Final numbers come from this Mac (the user ruled a 16-core machine optional; N5 never arranged).
- Compute-bound toy loop, `schedule(dynamic,1)`: 1.96× at 2 threads, 4.15× at 5, 6.30× at 11. Linear on the P-cores,
  then each E-core adds ~0.4 of a P-core (ceiling ≈ 7.4 P-core equivalents).
- No thread pinning on macOS (`omp_get_num_places()` is 0), no `perf` (use Instruments). Report threads 1–5 (P-cores)
  apart from 6–11 (E-cores join).
- **Benchmarks only on mains power** (`run_bench.py --require-ac`): on battery the Mac idle-sleeps mid-run even under
  `caffeinate -i`, and throttles in dark wake. Stalled rows (wall > 2 × CPU + 0.05 s) are dropped and counted.

## Decisions (don't re-litigate)
- C++20 + OpenMP; PCRE2 (RE2 lacks lookahead); `ankerl::unordered_dense`.
- **IncrSerial is the speedup baseline** (serial pre-tokenizer, 1 thread). Naive engines only on small configs.
- Chunking only at special-token boundaries (SPEC §2); tie-break on `(bytes(a), bytes(b))` (SPEC §4).
- TSan needs clang + LLVM libomp (libgomp gives false races); Archer's `ignore_noninstrumented_modules=1` in CI.
- Defaults from the B4 sweep: `owners_per_thread` 1, `par_threshold_tokens` 4096, `batch_scan_cap` 4096,
  `batch_skip_small` on (small rounds merge one pair; `--batch-full-rounds` for span figures).
- Every figure and report number uses one CSV with its own baseline (`bench/results.py`, `plot.py pick_sources`).

## Lessons worth knowing
- **Bucket by the low hash bits.** `ankerl` indexes its slots by the high bits; bucketing per-thread maps by the high
  bits made 3 threads 51× slower than serial (S4).
- **Push a new token's bytes before any heap push:** the heap comparator reads them. The out-of-bounds read broke 9/11
  golden cases and UBSan missed it; the bounds-checked Debug build traps it (S3).
- **PCRE2's `\s` matched U+180E**, Unicode 16 doesn't: `\s`/`\S` compile as `\p{White_Space}` (S1).
- **The batch formula needs a 0-based j**; a 1-based j keeps one member too many (`eax`×2, `df`×2). Wrong rules pass on
  wikitext.full, so exactness needs the small random corpora too (B0).
- **IncrPar forks only for big steps** (no persistent region): ~31k merges mostly touch a few words (S6).
- **Hashed owners fix the imbalance (2.30 → 1.08) but barely speed up IncrPar**: batching is what pays (B5).
- **11 threads merge slower than 5:** each round waits for an E-core. `--merge-threads auto` keeps phase 3 on the
  P-cores (B8).
- **TinyStories' 0.94× is a baseline effect** (the baseline row used the serial pre-tokenizer), not parallel overhead.
- **Core samples are weak for long events:** threads migrate within a 17 ms phase-1 unit (units that started on an
  E-core took as long as P-core ones). Phase-3 events are microseconds, so their P/E samples hold (B10).
- `HTTPServer.server_bind` does a reverse DNS lookup that stalled 35 s here; `studio/serve.py` skips it.

## Where things are
| Path | What it is |
|---|---|
| `SPEC.md` | The training spec and file formats: the contract for every engine and the Python reference |
| `include/parabpe/`, `src/` | Library and CLI; one file per strategy (`merge_naive`, `merge_incr`, `merge_batch`, `pretok_*`) |
| `src/incr_common.hpp`, `src/batch_select.hpp` | Owners, sharded counts, parallel steps; the batch rule |
| `src/trace*.{hpp,cpp}`, `src/sysctl.hpp` | `--trace`, `--trace-words`; macOS core counts |
| `tests/` | `unit/` (GoogleTest), `golden/` (11 cases), `equivalence/run_matrix.py`, `interop/`, `studio/`, `bench/` |
| `reference/` | Python oracle `bpe_ref.py`, `make_golden.py`, byte-format notes |
| `bench/` | `run_bench.py` (YAML → `results/*.csv`), `plot.py`, `results.py` (row rules), `kill_check.py` |
| `results/` | Benchmark CSVs, `plots/` + `summary.md`, profile and trace summaries |
| `report/` | `report.tex` → `report.pdf`; `numbers.py` → `numbers.tex` (every number from `results/`) |
| `studio/` | ParaBPE Studio (`serve.py` + viewer); guide in `docs/notes/studio.md` |
| `scripts/` | PCRE2 build, data download and slicing, profiling spike, `trace_summary.py`, `run_s8.sh` |
| `tools/batchsim/` | Independent batching simulator (no project code): the B0 cross-check |
| `docs/notes/` | Batched-merging proof, profiling, related work, Studio guide |

## Log (one line per task; details in git history)
| Date | Task | Outcome |
|---|---|---|
| 10-31 | N1, N2, N4 | Skeleton, CMake, CI; spec + interfaces; Python reference + 11 golden cases (Unicode 16 pinned) |
| 11-01 | N3 | `get_data.sh` + `slice_data.py`: TinyStories and WikiText-103 slices, checksummed |
| 11-02 | S1, S2 | Chunker + serial pre-tokenizer (golden `words.tsv`); WordTable, NaiveSerial, byte-exact writers |
| 11-03 | S3 | IncrSerial: 125–266× faster merge phase than NaiveSerial; equivalence matrix in CTest |
| 11-03 | S4, S10 | OMP pre-tokenizer (bucketed reduction, 7.0× at 11 threads on 1 GB); parallel chunker |
| 11-03 | S5, S6, S7 | Parallel pair stats; IncrPar (1.14× at 5 threads, fork per big step); NaivePar |
| 11-03 | N7, N8, G1 | Bench harness; profiling spike (phase 1 = 76–95% of serial time); modest-S6 decision |
| 11-03–05 | B0–B3 | Batch rule + proof + `batchsim`; refactor; `incr-batch` plumbing; serial batched engine + oracle |
| 11-07 | B4 | Hashed owners, parallel rounds; kill check passed (1.83× / 1.39× phase 3 at 5 threads) |
| 11-06 | B5, N6 | Owner-map ablation (load 2.30 → 1.08); related work (no prior exact parallel batching found) |
| 11-07 | S8 | All benchmarks on mains power: 2,232 rows, one hash per input; incr-batch 1.63× at 5 threads |
| 11-07 | B6, B7 | `--trace` (+1.2% at most); ParaBPE Studio |
| 11-08 | S9 | Report (tectonic); weak-scaling chart bug fixed; PCRE2 10.45 built locally |
| 11-09 | B8 | `--merge-threads N\|auto` (phase 3 on P-cores); mixed teams in the matrix |
| 11-13 | B9, B10 | Studio race (lockstep replay + core map); work view (units, real words per thread) |
| 11-18 | Cleanup | Docs, C++ and tooling pass; `bench/results.py`; outputs verified unchanged |
| 11-18 | B8 bench, S9 | `merge_threads.csv`: auto holds 1.70× at 11 threads; report §8.2; figures match text |

## Open
- Run-order test for the TinyStories phase-1 miss: `bench/configs/merge_threads_order.yaml` (mains power, ~3 min).
- SPEC.md and the interfaces were drafted by the agent; the team review (N2) never happened.
- Submission deadline: not recorded.
