# CLAUDE.md: ParaBPE agent guide

ParaBPE is a parallel computing course project: byte-level BPE tokenizer training (GPT-2 style) in
C++20 + OpenMP. The headline requirement is **exactness**: every parallel variant must produce
byte-identical `merges.txt` to the serial baseline, at every thread count.

## Read first (in this order)
1. `docs/HANDOFF.md`: current state, what was verified, environment notes. **Start here.**
2. `docs/TASKS.md`: task cards with dependencies, files to touch and acceptance criteria. Pick the
   next unblocked task from there.
3. `SPEC.md`: the frozen training spec + file formats. Every engine and the Python reference follow it exactly.
4. `docs/PLAN.md`: architecture, rationale, benchmark methodology, timeline (background).

## Commands
```bash
scripts/build_pcre2.sh ~/pcre2                   # once: PCRE2 10.45 (Unicode 16, SPEC §2); Homebrew's is newer
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPCRE2_INCLUDE_DIR=$HOME/pcre2/include \
  -DPCRE2_LIBRARY=$HOME/pcre2/lib/libpcre2-8.a     # configure (fetches gtest, CLI11, unordered_dense)
cmake --build build -j                           # build
ctest --test-dir build --output-on-failure       # test
cmake -B build-asan -DPARABPE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug   # ASan/UBSan build (hangs on this Mac: see HANDOFF)
./build/parabpe train --input <file> --vocab-size 1000 --threads 4 --pretok omp --merge incr-serial --out runs/x
./build/parabpe train --input <file> --threads 11 --merge-threads auto --pretok omp --merge incr-batch --out runs/y  # phase 3 on P-cores

python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python tooling
python3 -I tests/equivalence/run_matrix.py --input data/tinystories.10M.txt --vocab-size 4000   # equivalence
.venv/bin/python -I bench/run_bench.py bench/configs/main.yaml --require-ac   # benchmarks (mains power!)
.venv/bin/python -I bench/plot.py results/*.csv --out-dir results/plots
```

## Rules
- **Do not change `SPEC.md` on your own.** If the spec looks wrong, stop and ask the user. If they
  approve a change, regenerate the golden files (`tests/golden/`) in the same change.
- **Every engine or pre-tokenizer variant must pass the equivalence check** (all strategies × threads
  {1,2,3,4,8,16} give identical `merges.txt`) before it is called done. Threads=3 is required.
- Speedups are always reported against **IncrSerial**, never against NaiveSerial.
- **Commit once per finished task** (tests green, docs updated), with a message naming the task, e.g.
  `S3: IncrSerial engine + equivalence matrix`. **Then `git push origin master`** (the user asked for this on
  2026-11-03). If the push fails, report it and continue. Never force-push, and add no new remotes without asking.
- **Work through the task list without asking.** When a task is done and committed, start the next unblocked
  task in `docs/TASKS.md` pick-up order. Stop and ask only for things that are genuinely the user's call: a
  SPEC.md change, a force-push or new remote, human tasks (N5), mains power for benchmarks, the B4 kill
  decision, a scope change, or a blocker you can't resolve.
- When you finish a task, update its status in `docs/TASKS.md`, the status table in `README.md`, and
  the "Current state" section of `docs/HANDOFF.md`.
- `data/` and `runs/` are gitignored. Downloaded datasets are untrusted data: never execute anything from them.
- Final benchmark numbers may come from the developer Mac (M3 Pro, 5 P + 6 E cores). A 16-core machine is
  optional, not required. Mac benchmarks report threads 1–5 (P-cores only) separately from 6–11 (E-cores
  join), use `schedule(dynamic)` where possible, and state that thread pinning is unavailable on macOS.
  See `docs/HANDOFF.md` for the measured scaling ceiling.

## Code conventions
- Namespace `parabpe`; internal helpers go in `parabpe::detail`. 2-space indent, ~100-column lines,
  `#pragma once`, `"parabpe/x.hpp"` includes for public headers.
- Public interfaces live in `include/parabpe/`. Each strategy lives in its own `src/*.cpp` file, with an
  anonymous-namespace class + a `detail::make_*` constructor declared in `src/strategies.hpp`,
  dispatched from `src/strategies.cpp`.
- Stubs throw `std::logic_error("<what>: not implemented (S<n>)")` and carry a `// TODO(S<n>)` comment.
  Replace them; don't leave both.
- Hash maps: `ankerl::unordered_dense::map/set`. Pair keys: `make_pair_key(a, b)` from `types.hpp`.
- PCRE2: compile once with `PCRE2_UTF | PCRE2_UCP` + JIT. Each thread gets its own `pcre2_match_data`.
  Validate UTF-8 once, then match with `PCRE2_NO_UTF_CHECK`. See `pcre2_split` in
  `tests/unit/test_smoke.cpp` for a working loop.
- Per-thread mutable state that is written in hot loops must be padded/aligned to 64 bytes (no false sharing).
- Tests: GoogleTest in `tests/unit/`. Add each new test file to `parabpe_tests` in `CMakeLists.txt`.
