# ParaBPE: Parallel Byte Pair Encoding Tokenizer Training

ParaBPE trains a byte-level BPE tokenizer (GPT-2/Llama style) in parallel on multicore CPUs,
using C++20 and OpenMP. Every parallel variant must produce **exactly the same merges** as the
serial baseline, and speedup is measured against the best serial implementation.

- [SPEC.md](SPEC.md): the training spec (chunking, pair counting, tie-break, output format)
- [docs/PLAN.md](docs/PLAN.md): the architecture, milestones, benchmark methodology and timeline
- [docs/TASKS.md](docs/TASKS.md): the task board with acceptance criteria
- [STATE.md](STATE.md): where the project is right now (one screen); the place to start
- [docs/HANDOFF.md](docs/HANDOFF.md): dated per-task log and environment notes

## Build

Requirements: CMake ≥ 3.20, a C++20 compiler, OpenMP and **PCRE2 exactly 10.45**. The spec pins Unicode 16.0;
Homebrew's PCRE2 is newer (Unicode 17) and Ubuntu's is older, so build 10.45 once with the script:

```bash
# macOS: brew install cmake libomp     Ubuntu: sudo apt install cmake
scripts/build_pcre2.sh ~/pcre2
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DPCRE2_INCLUDE_DIR=$HOME/pcre2/include -DPCRE2_LIBRARY=$HOME/pcre2/lib/libpcre2-8.a
cmake --build build -j
ctest --test-dir build --output-on-failure
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python tooling, Studio, benchmarks
```

Sanitizer builds: `-DPARABPE_ASAN=ON` or `-DPARABPE_TSAN=ON` (TSan needs clang with LLVM libomp). Both run in CI; on
the developer Mac (macOS 27) the local ASan and TSan runtimes hang at startup, so use UBSan there (docs/HANDOFF.md).

## Usage

```bash
scripts/get_data.sh wikitext       # data/wikitext.{10M,100M,full}.txt
./build/parabpe train --input data/wikitext.full.txt --vocab-size 32000 --threads 11 \
    --merge-threads auto --pretok omp --merge incr-batch --out runs/wikitext-32k
```

| Flag | Values |
|---|---|
| `--pretok` | `serial`, `omp` |
| `--merge` | `naive-serial`, `incr-serial`, `naive-par`, `incr-par`, `incr-batch` |
| `--merge-threads` | `N` or `auto`: thread count for phase 3 (merging); default = `--threads`. `auto` = the performance cores, at most `--threads` (5 on the M3 Pro), so phase 1 can use every core while merging stays off the E-cores |
| `--par-threshold-tokens` | IncrPar, incr-batch: minimum tokens touched in a step before the work is split across threads |
| `--verify-batches` | incr-batch: check every member against a sequential oracle |
| `--batch-full-rounds` | incr-batch: batch every round, even ones too small to go parallel (the span figures) |
| `--trace-words FILE` | The 400 most frequent words (4+ bytes) with their phase-3 owner, as JSON (for Studio) |
| `--trace FILE` | Chrome Trace Event JSON of phases 1 and 3 per thread, with the sampled CPU and P/E core; open in ui.perfetto.dev, summarise with `scripts/trace_summary.py` |

## ParaBPE Studio

```bash
python3 -I studio/serve.py      # then open http://127.0.0.1:8765/
```

Trains a run with `--trace` on a corpus in `data/`, replays it per thread with P/E-core samples, shows each round's
merges landing and the span curve, checks `merges.txt` against an IncrSerial reference (sha256), plots
`results/*.csv`, and encodes a sentence with Hugging Face `tokenizers` (needs `.venv`). **The race** replays two or
three recorded runs side by side on one clock (IncrSerial on 1 thread against incr-batch with `--merge-threads auto`),
with a core map of the M3 Pro lighting up per lane; what it shows and what it doesn't is in
[docs/notes/race.md](docs/notes/race.md). **The work** shows the same replay on the data: the corpus's work units filling in, then real words of the run fusing round by round, grouped by the thread that rewrote them ([docs/notes/work_view.md](docs/notes/work_view.md)). The server listens on 127.0.0.1 only and accepts only `*.txt` files
directly under `data/`. The demo script is in [docs/notes/demo.md](docs/notes/demo.md).

## Layout

```
include/parabpe/   public interfaces (types, corpus, regex, pretokenizer, merge_engine, trainer, io)
src/               implementations, one file per strategy
reference/         Python oracle (bpe_ref.py), the spec-exact reference for tiny corpora
tests/             unit/, equivalence/ (strategy × threads), golden/, interop/ (Hugging Face), studio/
bench/             run_bench.py, plot.py, configs/
results/           benchmark CSVs and plots (committed)
scripts/           PCRE2 build, data download + slicing, profiling spike, trace_summary.py
tools/batchsim/    independent batching simulator (the B0 cross-check)
studio/            ParaBPE Studio (serve.py + a no-build viewer)
report/            LaTeX report (report.tex, report.pdf) and numbers.py
docs/              plan, task board, handoff log, notes
```

## Status

All tasks are done except the B8 benchmark (needs mains power) and the report's final review. Per-task
status: [docs/TASKS.md](docs/TASKS.md); where things stand now: [STATE.md](STATE.md).
