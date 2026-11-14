# The race (Studio, B9): what it shows and why it is honest

The Race card in ParaBPE Studio replays two or three *recorded* training runs side by side on one clock. It exists
to make one point visible: what parallelism bought, phase by phase, on this machine. This note explains every element,
the concept behind it, and what the picture does **not** show. Code: `studio/studio.js` (section "the race"),
`studio/serve.py` (`/api/runs`, `merge_threads`).

## The three rules that make it believable
1. **Nothing is animated from a formula.** Each lane is a `trace.json` written by `parabpe train --trace`: Chrome
   Trace Event JSON with real wall-clock timestamps (µs from the start of the run). Studio parses it
   (`parseTrace`) and advances one shared clock `t`; every lane shows its state at that same `t`.
2. **Fair comparison.** Lane A is IncrSerial on one thread with the serial pre-tokenizer: the baseline every speedup
   in this project is measured against (CLAUDE.md, report). The other lanes train the **same corpus to the same
   vocabulary**, so they do the same work. Lanes B and C can only be chosen from runs on lane A's corpus and vocab.
3. **The audience can make it live.** "Train race set" trains the lanes in front of them (about 10 s per lane on
   `wikitext.full`), then replays. Nothing is cached except IncrSerial's reference sha256 per corpus and vocab.

## Reading a lane
| Element | What it is | What it teaches |
|---|---|---|
| Progress bar + counter | merges landed by `t`, out of the vocabulary's merge count | the merge chain is the work that does not parallelise by itself |
| Phase label | phase 1 (chunk + pre-tokenize), 2 (word table + pair counts), 3 (merges), done | Amdahl's law: phase 1 scales with threads, phase 3 only with batching |
| Clock | the lane's own elapsed wall-clock time; at the finish, its total | wall-clock, not CPU time: 11 busy cores for 1 s cost 11 CPU-seconds |
| Core map (P row, E row) | one tile per CPU of the M3 Pro: 5 performance cores (6–10) and 6 efficiency cores (0–5); a tile is lit when a thread of this lane ran a traced event there in the last 0.5 % of the race clock, coloured by the work kind (the swimlane legend) | which cores are doing work; with `--merge-threads auto` the E tiles go dark in phase 3 |
| Dashed tile | the lane is inside a measured phase but no event is traced: one thread is working | the serial pre-tokenizer and the serial merge engines are timed, not traced per event |
| 🏁 and "N× faster than A" | the first lane to reach its end, and every lane's total against lane A's | the total-time speedup, the same number as `results/` reports |
| Badge | all lanes' `merges.txt` have the same sha256 | **exactness**: the parallel runs produced the byte-identical merge list |

**Merges landed.** For `incr-batch`, rounds are traced: `SELECT i` picks round *i*'s merges and they land when its
`ORDER_UPDATE` ends, so the counter jumps by the round size (up to 59 merges at once on WikiText). IncrSerial and
IncrPar are timed as a whole: their counter moves evenly across the measured phase-3 time, and the card says so.

## What a master's student should take from it
- **Phase 1 is embarrassingly parallel.** Lane B lights all 11 tiles in blue; lane A lights one. The bars do not
  move yet: no merge has happened in either lane.
- **Phase 3 is a dependency chain.** Lane A's merge count creeps; lane B's jumps in rounds. Batched rounds are the
  project's contribution: a round is a prefix of token-disjoint pairs, cut where a tie could reorder it, and it is
  *proved* equal to the next *k* sequential merges (`docs/notes/batched_merging.md`).
- **Heterogeneous cores.** With `--merge-threads auto` (B8), lane B merges on the P-cores only: the E tiles go dark
  in phase 3, and each round no longer waits for an efficiency core. Lane C (same threads, merges on all cores) shows
  the E tiles busy and finishes later.
- **Speedup is a wall-clock ratio against the serial baseline.** On WikiText-103 at vocabulary 32k (mains power,
  `results/batch.csv`): IncrSerial 5.65 s total; incr-batch at 11 threads 3.91× faster. The race shows the same
  ratio at whatever speed factor you pick, because both lanes are scaled by the same factor.

## What it does not show
- **Thread migration.** macOS cannot pin threads. The CPU is sampled once, at each event's start; a long event may
  move cores. Tiles are samples, not placements.
- **The gaps between traced events.** Heap updates, pair-count bookkeeping and phase 2 are timed but not traced per
  event. That is why the core map looks back over a small window and why the dashed tile exists.
- **Memory bandwidth and cache effects**, which are a large part of why phase 1 scales 5.1× and not 11× at 11 threads.
- **Measurement hygiene.** A race set trained on battery or with other programs running is still a real trace,
  just a slower one. For numbers that go in the report, use `bench/run_bench.py --require-ac`.

## Stills for slides
`http://127.0.0.1:8765/#race&at=0.3` opens on the Race card 30 % of the way through the slowest lane;
`#race&end` shows the finish. Add `&light` or `&dark` to force a theme.
