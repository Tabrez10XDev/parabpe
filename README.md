# ParaBPE: Parallel Byte Pair Encoding Tokenizer Training

ParaBPE trains a byte-level BPE tokenizer (GPT-2/Llama style) in parallel on multicore CPUs,
using C++20 and OpenMP. Every parallel variant must produce **exactly the same merges** as the
serial baseline, and speedup is measured against the best serial implementation.

- [SPEC.md](SPEC.md): the frozen training spec (chunking, pair counting, tie-break, output format)
- [docs/PLAN.md](docs/PLAN.md): the architecture, milestones, benchmark methodology and timeline
- [docs/TASKS.md](docs/TASKS.md): the task board with acceptance criteria
- [docs/HANDOFF.md](docs/HANDOFF.md): current state and environment notes, written for whoever (human or agent) picks up next

## Build

Requirements: CMake ≥ 3.20, a C++20 compiler, OpenMP and PCRE2.

```bash
# macOS
brew install cmake libomp pcre2
# Ubuntu
sudo apt install cmake libpcre2-dev

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Sanitizer builds: `-DPARABPE_ASAN=ON`, or `-DPARABPE_TSAN=ON` (use clang with LLVM libomp for TSan).

## Usage

```bash
./build/parabpe train --input data/wt103.txt --vocab-size 32000 --threads 16 \
    --pretok omp --merge incr-par --out runs/wt103-16t
```

| Flag | Values |
|---|---|
| `--pretok` | `serial`, `omp` |
| `--merge` | `naive-serial`, `incr-serial`, `naive-par`, `incr-par`, `incr-batch` |
| `--par-threshold-tokens` | IncrPar, incr-batch: minimum tokens touched in a step before the work is split across threads |
| `--verify-batches` | incr-batch: check every member against a sequential oracle |
| `--batch-full-rounds` | incr-batch: batch every round, even ones too small to go parallel (the span figures) |

## Layout

```
include/parabpe/   public interfaces (types, corpus, pretokenizer, merge_engine, io)
src/               implementations, one file per strategy
reference/         Python oracle (bpe_ref.py), the spec-exact reference for tiny corpora
tests/             unit/, equivalence/ (strategy × threads), golden/
bench/             run_bench.py, plot.py, configs/
scripts/           data download + slicing, profiling spike
report/            LaTeX report
```

## Status

| Step | Work | Status |
|---|---|---|
| N1 | Repo skeleton, CMake, CI, smoke tests | ✅ |
| N2 | Spec + interfaces frozen | ✅ (draft; team review pending) |
| N3 | Data scripts | ✅ |
| N4 | Python reference + golden files | ✅ |
| S1 | Chunker + serial pre-tokenizer | ✅ |
| S2 | WordTable + NaiveSerial + writer | ✅ |
| S3 | IncrSerial (baseline) | ✅ |
| S4 | OMP pre-tokenizer | ✅ |
| S5 | Parallel PairStats | ✅ |
| S6 | IncrPar | ✅ |
| S7 | NaivePar | ✅ |
| S10 | Parallel chunker | ✅ |
| B0 | Batched merging: proof, `tools/batchsim` | ✅ |
| B1 | Refactor: `src/incr_common.hpp` | ✅ |
| B2 | `incr-batch` plumbing | ✅ |
| B3 | `incr-batch` serial engine + `--verify-batches` | ✅ |
| B4 | Hashed owners, parallel rounds | 🟨 kill check pending (mains power) |
| B5 | Owner-map ablation | ✅ |
| N6 | Related work | ⬜ |
| S8 | Benchmarks | ⬜ |
| B6–B7 | Trace, ParaBPE Studio | ⬜ |
| S9 | Report | ⬜ |
