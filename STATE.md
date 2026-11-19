# STATE: where the project is right now

Updated 2026-11-18 · last commit: S9 (figures use one CSV + own baseline; report review) · check `gh run list`

> One screen. **Overwrite, don't append**: history goes to `docs/HANDOFF.md` (dated log). Max ~60 lines.
> Every agent updates this file before ending a session or committing a task.

## Now
- **Every task in the pick-up order is done** (B0–S10, B8–B10). Every engine is byte-identical to IncrSerial at
  threads {1,2,3,4,8,16}; all 2,736 benchmark rows have one merges.txt hash per input and vocabulary.
- Report: `report/report.pdf` (9 pages, `cd report && tectonic report.tex`), every number from `report/numbers.py`.
  §8.2 has the `--merge-threads auto` results (HANDOFF "B8 benchmark + S9 update": 2 of 3 predictions met).
- Studio has Replay, **the work** (B10) and **the race** (B9); demo script `docs/notes/demo.md`.
- Cleanup pass done (HANDOFF "Cleanup"): `bench/results.py` holds the benchmark row rules for plots, report and checks.

## Next
1. On **mains power** (~3 min): `.venv/bin/python -I bench/run_bench.py bench/configs/merge_threads_order.yaml
   --require-ac`. It runs `auto` first. If phase 1 is now slower *without* auto, the B8 TinyStories miss was run order:
   say so in report §8.2 (last paragraph) and HANDOFF "B8 benchmark"; if auto is still slower, report that instead.

## Waiting on the user
- Mains power for step 1 (on battery as of 2026-11-18 evening).
- Student: rehearse the demo twice (dry run of every step passed); N5 (big machine, optional); deadline unknown.

## Traps (read before touching anything)
- Never edit `SPEC.md`. Speedups are vs **IncrSerial**. Benchmarks only with `--require-ac`.
- PCRE2 must be 10.45 from `~/pcre2` (Homebrew's is Unicode 17 and breaks the pin test).
- ASan and local TSan hang on this Mac: use CI for them. Always `.venv/bin/python -I`.
- Commit per task, then `git push origin master`; never force-push.

## Index: need → read
| Need | Read |
|---|---|
| Rules, commands, code conventions | `CLAUDE.md` |
| The contract (algorithm, tie-break, file formats) | `SPEC.md` |
| Task cards, acceptance criteria, status table | `docs/TASKS.md` (`### <task id>`) |
| Why a past decision was made / per-task log | `docs/HANDOFF.md` (`### <task id> (<date>)`) |
| Decisions not to re-litigate | `docs/HANDOFF.md#decisions-already-made-dont-re-litigate` |
| Toolchain, PCRE2, ASan/TSan, Python env | `docs/HANDOFF.md#environment-notes-developer-mac` |
| P/E-core scaling ceiling, bench hygiene | `docs/HANDOFF.md#benchmark-machine-decided-2026-10-31` |
| Architecture, benchmark methodology | `docs/PLAN.md` |
| Batched merging (B0–B5): proof and risks | `docs/notes/batched_merging.md` |
| Profiling findings / related work / demo script | `docs/notes/{profiling,related_work,demo}.md` |
| Studio race: what each element shows, what it doesn't | `docs/notes/race.md` |
| Studio work view (units, words per thread), trace `unit` + `--trace-words` | `docs/notes/work_view.md` |
| Benchmark results and plots | `results/*.csv`, `results/plots/`; report numbers via `report/numbers.py` |
| Which benchmark rows count (warm-ups, stalls, phase 1) | `bench/results.py` |
| Where code lives | `docs/HANDOFF.md#what-exists`; engines are `src/merge_*.cpp`, pretok `src/pretok_*.cpp` |
