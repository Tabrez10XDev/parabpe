# ParaBPE Studio demo (3 minutes)

Setup, before the audience arrives:

```bash
cmake --build build -j
python3 -I studio/serve.py            # http://127.0.0.1:8765/ (local only)
```

Plug the Mac into mains power. In the Studio, train these two runs once so the IncrSerial reference sha256 is
cached (about 10 s each):
- `wikitext.full.txt`, vocab 32,000, 11 threads, incr-batch, with **batch every round** ticked (the span run);
- the same without the tick (the fast default run).

`http://127.0.0.1:8765/#end&p3` opens the newest run already finished, on the phase 3 view: a still for the slides.
Add `&light` or `&dark` to force a theme.

All numbers below are measured (`results/`, mains power, wikitext.full). Where the mission's draft script differed, the
measured value is used and the difference is listed at the end.

## Script

1. **Amdahl** (Benchmarks, corpus wikitext.full, vocab 32,000). "Pre-tokenizing scales: 4.0× at 5 threads and 5.1× at
   11. The merge chain doesn't: IncrSerial's phase 3 stays at 1.0×. Every merge depends on the one before it."
2. **Replay** (the span run, View: whole run, Play at "fit to 20 s"). "Eleven OpenMP threads. Blue is pre-tokenizing in
   parallel. Then phase 3: the master selects a batch (yellow), the owners rewrite their words in parallel (green)."
   Point at the P/E badges: "macOS can't pin threads, so these are samples: where each thread was when its work began."
3. **Batches** (switch View to phase 3, Play). "31,743 merges in 2,561 rounds, up to 59 merges in one round: 12.4×
   fewer sequential steps." Point at a big round landing in *Merges landing*.
4. **Exactness badge** turns green at the end: "merges.txt is byte-identical to IncrSerial's. Same sha256."
5. **Theorem** card. "A batch is a best-first prefix of token-disjoint pairs, cut where a tie could reorder it. From
   any state, sequential BPE would make exactly these k merges next, in this order. Proof in the notes; every member is
   also checked at runtime by `--verify-batches`, and the test matrix compares all engines at 1, 2, 3, 4, 8 and 16
   threads."
6. **Before/after** (Phase 3 chart, then Owner load). "IncrPar, one merge per step: 1.15× at 5 threads. incr-batch:
   1.63× (1.31× at vocab 100k). Hashing words to owners cuts the busiest owner from 2.30× the mean to 1.08." Also: "on
   the E-cores it drops (1.38× at 11 threads), because each round waits for the slowest owner."
7. **The race** (Race card: "Train race set" with the Train form at wikitext.full, vocab 32,000, 11 threads, ~15 s
   for both lanes; or load the two runs trained before the talk). Play at "fit to 20 s". "Lane A is IncrSerial on one
   thread, the baseline every number in this project is measured against. Lane B is incr-batch on 11 threads with the
   merges on the five performance cores. Same corpus, same vocabulary; each lane runs at its own recorded speed."
   Point at the core maps: "Phase 1 lights every core. In phase 3 the E-core tiles go dark: the merge team is the five
   P-cores, so no round waits for an efficiency core." At the flag: "3.9× on the whole run (measured: `results/`),
   and the badge says both wrote the same `merges.txt`." Tick "add lane C" for the before/after of
   `--merge-threads auto` at 11 threads. What the picture does and doesn't show: `docs/notes/race.md`.
8. **Hugging Face** (Load into tokenizers, Encode). "The merges load into HF `tokenizers` and give the same token ids
   as our Python reference encoder." In a terminal, the long version:
   `.venv/bin/python -I tests/interop/check_hf.py --model runs/studio/<id> --heldout data/wikitext.10M.txt`.

## Differences from the draft script (report, don't hide)

| Draft said | Measured | Source |
|---|---|---|
| phase 1 scales 7× | wikitext.full: 5.1× at 11 threads, 4.0× at 5. tinystories.1G: 6.4× at 11 (7.0× was an earlier single run, S4) | `results/batch.csv`, v32k |
| 31,743 merges / 3,409 rounds | 2,561 rounds with every round batched; 19,888 in the default mode, which merges small rounds one pair at a time because they can't go parallel | `results/span.csv`, `tools/batchsim` agrees |
| bursts of up to ~59 merges | 59 (span run), 58 (default run) | `results/span.csv`, `timings.json` |

The default run is the fast one (phase 3 1.63× at 5 threads); the span run shows the theorem's reach. Don't quote a
round count from one mode next to a speedup from the other.
