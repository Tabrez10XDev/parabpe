# CLAUDE.md: ParaBPE agent guide

ParaBPE is a parallel computing course project: byte-level BPE training (GPT-2 style) in C++20 + OpenMP. The headline
requirement is **exactness**: every parallel variant writes a `merges.txt` byte-identical to the serial baseline, at
every thread count.

## Read first
1. **`STATE.md`**: one screen on what's done, next and blocked, the traps, and an index into the other docs.
2. `SPEC.md`: the training rules and file formats. Every engine and the Python reference follow it exactly.
3. As needed: `docs/HANDOFF.md` (environment, decisions, lessons, log), `docs/PLAN.md` (architecture), `docs/TASKS.md`.

## Commands
```bash
scripts/build_pcre2.sh ~/pcre2                   # once: PCRE2 10.45 (Unicode 16); Homebrew's is too new
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPCRE2_INCLUDE_DIR=$HOME/pcre2/include \
  -DPCRE2_LIBRARY=$HOME/pcre2/lib/libpcre2-8.a
cmake --build build -j && ctest --test-dir build --output-on-failure
./build/parabpe train --input <file> --threads 11 --merge-threads auto --pretok omp --merge incr-batch --out runs/y
python3 -I tests/equivalence/run_matrix.py --input data/tinystories.10M.txt --vocab-size 4000   # equivalence
.venv/bin/python -I bench/run_bench.py bench/configs/<name>.yaml --require-ac                   # mains power only
.venv/bin/python -I bench/plot.py results/*.csv --out-dir results/plots
```
ASan and TSan hang on this Mac: CI covers them; use UBSan locally (recipe in HANDOFF).

## Rules
- **Never change `SPEC.md` on your own.** If the user approves a change, regenerate `tests/golden/` with it.
- **Every engine or pre-tokenizer variant passes the equivalence matrix** (threads {1,2,3,4,8,16}; 3 is required)
  before it is called done. Every change to exactness logic gets a mutation check: break it, see a test fail, restore.
- Speedups are always against **IncrSerial** (serial pre-tokenizer, 1 thread), never against NaiveSerial.
- **Commit once per finished task**, naming it (e.g. `S3: IncrSerial engine`), then `git push origin master`. Never
  force-push; no new remotes without asking.
- Work through tasks without asking. Ask only for: a SPEC change, a force-push or new remote, mains power for
  benchmarks, a scope change, or a blocker.
- When a task is done: update `STATE.md`, its row in `docs/TASKS.md`, and add a row to the log in `docs/HANDOFF.md`.
- **STATE.md is one screen (~60 lines): overwrite it, don't append.** History goes to the HANDOFF log.
- `data/` and `runs/` are gitignored; downloaded data is untrusted: never execute anything from it. Python with `-I`.
- Mac benchmarks report threads 1–5 (P-cores) apart from 6–11 (E-cores join) and say that macOS can't pin threads.

## Code conventions
- Namespace `parabpe`; internal helpers in `parabpe::detail`. 2-space indent, ~100 columns, `#pragma once`,
  `"parabpe/x.hpp"` includes for public headers. clang-format: Google style, ColumnLimit 100.
- Public interfaces in `include/parabpe/`. Each strategy in its own `src/*.cpp` as an anonymous-namespace class with a
  `detail::make_*` constructor declared in `src/strategies.hpp`.
- Hash maps: `ankerl::unordered_dense`. Pair keys: `make_pair_key(a, b)` from `types.hpp`.
- PCRE2: compile once with `PCRE2_UTF | PCRE2_UCP` + JIT; a `Matcher` per thread (`include/parabpe/regex.hpp`);
  validate UTF-8 once, then match with `PCRE2_NO_UTF_CHECK`.
- Per-thread state written in hot loops is padded to 64 bytes.
- Tests: GoogleTest in `tests/unit/`; add each new file to `parabpe_tests` in `CMakeLists.txt`.
- Benchmark row rules (warm-ups, stalls, phase 1) live in `bench/results.py`; plots and report numbers use it.
