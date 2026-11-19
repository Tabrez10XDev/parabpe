# ParaBPE: exact parallel BPE tokenizer training

[![CI](https://github.com/Tabrez10XDev/parabpe/actions/workflows/ci.yml/badge.svg)](https://github.com/Tabrez10XDev/parabpe/actions/workflows/ci.yml)

ParaBPE trains a byte-level BPE tokenizer (GPT-2 style) on a multicore CPU with C++20 and OpenMP. Every parallel
variant produces a `merges.txt` and `vocab.json` that are **byte-for-byte identical** to the serial baseline, at every
thread count.

The merge phase of BPE looks inherently serial: each merge depends on the one before. ParaBPE's `incr-batch` engine
merges several pairs per parallel round, and a proof shows each round equals the next steps of sequential BPE,
including order and tie-breaking.

**Results** (Apple M3 Pro, 5 performance + 6 efficiency cores; WikiText-103, 540 MB, vocabulary 32k; median of 5,
against the serial incremental trainer):

| | Speedup |
|---|---|
| Merge phase, `incr-batch`, 5 threads | **1.63×** (one merge per step: 1.15×) |
| Merge phase, 11 threads with `--merge-threads auto` | **1.70×** (without: 1.47×) |
| Whole run, 11 threads | **3.91×** |
| Sequential steps with batched rounds | 2,561 rounds for 31,743 merges (**12.4× fewer**) |

The full write-up is in [report/report.pdf](report/report.pdf).

## Quick start

### 1. Install the tools
- **macOS:** `brew install cmake libomp` (Xcode command-line tools provide clang)
- **Ubuntu:** `sudo apt install cmake g++ curl`

You also need Python 3.11+ for the tests, benchmarks and the Studio viewer.

### 2. Build PCRE2 10.45 (once)
The spec pins Unicode 16, which needs exactly PCRE2 10.45. Homebrew's version is too new and Ubuntu's too old, so the
repo builds its own copy:
```bash
scripts/build_pcre2.sh ~/pcre2
```

### 3. Build and test
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DPCRE2_INCLUDE_DIR=$HOME/pcre2/include -DPCRE2_LIBRARY=$HOME/pcre2/lib/libpcre2-8.a
cmake --build build -j
ctest --test-dir build --output-on-failure
```
The first configure downloads GoogleTest, CLI11 and unordered_dense, so it needs network access. All tests pass
without any dataset; the equivalence matrix is skipped until `data/` exists (step 5).

### 4. Train on a bundled sample (no download)
```bash
./build/parabpe train --input tests/golden/unicode/input.txt --vocab-size 300 \
  --threads 4 --pretok omp --merge incr-batch --out runs/sample
cat runs/sample/merges.txt runs/sample/timings.json
```
Run it again with `--merge incr-serial --out runs/sample-serial`: `merges.txt` is identical.

### 5. Train on a real corpus
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/get_data.sh wikitext        # downloads WikiText-103 (a few hundred MB)
.venv/bin/python -I scripts/slice_data.py --source wikitext --sizes 10M,100M
./build/parabpe train --input data/wikitext.full.txt --vocab-size 32000 \
  --threads 11 --merge-threads auto --pretok omp --merge incr-batch --out runs/wikitext-32k
```
Use your machine's core count for `--threads`. `--merge-threads auto` runs the merge phase on the performance cores
only (on Apple Silicon); elsewhere it uses every thread. `scripts/get_data.sh tinystories` fetches TinyStories (2.2 GB).

### 6. Watch it in ParaBPE Studio (optional)
```bash
python3 -I studio/serve.py        # then open http://127.0.0.1:8765/
```
Train a run from the page and replay what every thread did: per-thread swimlanes, the corpus filling in work unit by
work unit, real words merging round by round, a side-by-side race against the serial trainer, and an exactness badge.
Guide and demo script: [docs/notes/studio.md](docs/notes/studio.md).

## Command-line options

`parabpe train --input FILE --out DIR [options]`, or `./build/parabpe train --help` for the full list.

| Option | Values |
|---|---|
| `--vocab-size` | target vocabulary size (default 32000) |
| `--threads` | threads for pre-tokenizing (and merging, unless `--merge-threads`) |
| `--merge-threads` | `N` or `auto` (the performance cores) for the merge phase |
| `--pretok` | `serial`, `omp` |
| `--merge` | `naive-serial`, `incr-serial` (the baseline), `naive-par`, `incr-par`, `incr-batch` |
| `--verify-batches` | `incr-batch`: check every merge against a sequential engine while training |
| `--trace FILE` | per-thread event trace (Chrome Trace JSON; open in ui.perfetto.dev) |

Outputs: `merges.txt` and `vocab.json` (the GPT-2 formats, loadable by Hugging Face `tokenizers`) and `timings.json`
(phase times, configuration, hardware).

## How exactness is checked
- **Golden files:** 11 small corpora whose outputs come from an independent Python reference (`reference/bpe_ref.py`).
- **Equivalence matrix:** every pre-tokenizer × engine at 1, 2, 3, 4, 8 and 16 threads on two 10 MB corpora must hash
  identically (`tests/equivalence/run_matrix.py`).
- **Runtime oracle:** `--verify-batches` replays each batched round against sequential BPE.
- **Benchmarks:** every one of the 2,736 benchmark runs records its `merges.txt` hash; each corpus and vocabulary has one.
- **Interop and sanitizers:** the output loads into Hugging Face `tokenizers` with matching token ids; CI runs GCC,
  AddressSanitizer and ThreadSanitizer on every push.

## Reproduce the benchmarks and report
On mains power, nothing else running:
```bash
# one grid; the others are in bench/configs/
.venv/bin/python -I bench/run_bench.py bench/configs/batch.yaml --require-ac
.venv/bin/python -I bench/plot.py results/*.csv --out-dir results/plots
.venv/bin/python -I report/numbers.py && (cd report && tectonic report.tex)
```

## Repository layout
```
include/parabpe/   public API headers
src/               implementation, one file per strategy (pretok_*, merge_*)
tests/             unit/ (GoogleTest), golden/, equivalence/, interop/, studio/, bench/
reference/         Python reference trainer (the oracle) and golden-file generator
bench/             benchmark runner, plots, shared row rules, configs/
results/           benchmark CSVs and plots
report/            report.tex, report.pdf, numbers.py
studio/            ParaBPE Studio (local server + viewer)
scripts/           PCRE2 build, data download and slicing, profiling, trace summary
tools/batchsim/    independent batching simulator (cross-check of the batch rule)
docs/              PLAN (architecture), TASKS, HANDOFF (environment, log), notes/
```

## Documentation
- [SPEC.md](SPEC.md): the training rules and file formats every engine follows
- [docs/PLAN.md](docs/PLAN.md): architecture, engines, verification and benchmark method
- [docs/notes/batched_merging.md](docs/notes/batched_merging.md): the batch rule and its proof
- [docs/notes/studio.md](docs/notes/studio.md): Studio guide and demo script
- [docs/HANDOFF.md](docs/HANDOFF.md) and [docs/TASKS.md](docs/TASKS.md): environment notes, lessons, project log, task status

## Notes
- macOS can't pin threads, so results report 1–5 threads (performance cores) apart from 6–11 (efficiency cores join).
- AddressSanitizer and ThreadSanitizer hang at startup on macOS 27 with Apple clang; they run in CI on Ubuntu.
- Datasets are not included (`data/` is created by the scripts).
