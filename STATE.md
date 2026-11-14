# STATE: where the project is right now

Updated 2026-11-13 · last commit `eb3324c` (B8) · CI green

> One screen. **Overwrite, don't append**: history goes to `docs/HANDOFF.md` (dated log). Max ~60 lines.
> Every agent updates this file before ending a session or committing a task.

## Now
- All tasks B0–S9 + B8 done. Every engine is byte-identical to IncrSerial at threads {1,2,3,4,8,16}.
- `--merge-threads N|auto` (B8) is in: phase 3 on the 5 P-cores while phase 1 uses all 11.
- Report draft builds: `report/report.pdf` (`cd report && tectonic report.tex`).
- **Uncommitted:** TinyStories correction in `report/{numbers.py,numbers.tex,report.tex}` (the 0.94× is a
  baseline-row effect, not parallel overhead; real loss is at 6–11 threads).

## Next (in order)
1. On **mains power**: `.venv/bin/python -I bench/run_bench.py bench/configs/merge_threads.yaml --require-ac`
   (~20 min), then `bench/plot.py results/*.csv --out-dir results/plots`.
2. Check the B8 predictions (HANDOFF#B8): t11+auto phase 3 within 5% of t5; TinyStories merge ≈ 51 ms. Report misses.
3. S9 update: numbers.py macros + `--merge-threads` paragraph in report.tex; fix HANDOFF TinyStories lines and
   `docs/notes/demo.md`. Commit `S9: report adds merge-on-P-cores results; TinyStories explained`.

## Waiting on the user
- Mains power for benchmarks (Mac on battery as of 2026-11-13).
- Student's review of the report draft; N5 (big machine, optional); submission deadline unknown.

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
| Benchmark results and plots | `results/*.csv`, `results/plots/`; report numbers via `report/numbers.py` |
| Where code lives | `docs/HANDOFF.md#what-exists`; engines are `src/merge_*.cpp`, pretok `src/pretok_*.cpp` |
