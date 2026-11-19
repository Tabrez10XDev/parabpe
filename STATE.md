# STATE: where the project is right now

Updated 2026-11-19 · project complete; submission zip built from the last commit.
One screen, overwritten (not appended) by whoever works next; history goes to the log in `docs/HANDOFF.md`.

## Now
- Every task is done except N5 (optional). Every engine is byte-identical to IncrSerial at threads
  {1,2,3,4,8,16}; all 2,736 benchmark rows have one `merges.txt` hash per input and vocabulary.
- Report: `report/report.pdf` (every number from `report/numbers.py`). Studio: Replay, the work view, the race.
- Submission: `git archive` of HEAD (see README); builds and passes its tests from a clean unpack.

## Optional next
- Run-order test for the TinyStories phase-1 miss (report §8.2): on mains power,
  `.venv/bin/python -I bench/run_bench.py bench/configs/merge_threads_order.yaml --require-ac` (~3 min; auto runs first).

## Traps
- Never edit `SPEC.md`. Speedups are vs **IncrSerial**. Benchmarks only with `--require-ac`.
- PCRE2 must be 10.45 from `~/pcre2` (Homebrew's is Unicode 17). ASan/TSan only in CI. Python: `.venv/bin/python -I`.
- Commit per task, then `git push origin master`; never force-push.

## Index
| Need | Read |
|---|---|
| Build, run, options, layout | `README.md` |
| Rules, commands, code conventions (agents) | `CLAUDE.md` |
| Training rules and file formats | `SPEC.md` |
| Architecture, engines, verification, benchmark method | `docs/PLAN.md` |
| Environment, decisions, lessons, file map, task log | `docs/HANDOFF.md` |
| Task status | `docs/TASKS.md` |
| Batch rule and proof | `docs/notes/batched_merging.md` |
| Studio guide and demo script | `docs/notes/studio.md` |
| Profiling, related work | `docs/notes/profiling.md`, `docs/notes/related_work.md` |
| Benchmark row rules; results | `bench/results.py`; `results/*.csv`, `results/plots/` |
