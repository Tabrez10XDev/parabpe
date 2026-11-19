# ParaBPE Studio: guide and demo script

Studio is a local web page that trains a run with `--trace` and replays what every thread did, on a timeline and on
the data. Every picture comes from a recorded trace: nothing is animated from a formula.

```bash
cmake --build build -j
python3 -I studio/serve.py        # open http://127.0.0.1:8765/ (listens on 127.0.0.1 only)
```

## The cards
| Card | Shows |
|---|---|
| **Train** | Runs `parabpe train --trace --trace-words` on a corpus in `data/`, plus a cached IncrSerial reference run for the exactness badge. |
| **Replay** | One lane per OpenMP thread: pre-tokenizing (blue), selection (yellow), owners rewriting words (green). P/E badges are core *samples* (macOS can't pin threads). The exactness badge compares `merges.txt` against IncrSerial by sha256. |
| **The work** | The same clock on the data. Phase 1: one square per ~1 MB work unit, filling in as threads take the next unit (`schedule(dynamic,1)`). Phase 3: real words of the run, rewritten round by round, in the lane of the thread that ran their owner; "Next round" steps through. |
| **The race** | Two or three recorded runs on one clock. Lane A is the baseline (IncrSerial, 1 thread); others must use the same corpus and vocabulary. Each lane has a progress bar, clock and a core map (5 P tiles, 6 E tiles). "Train race set" trains A and B live (~8 s on WikiText at 32k). |
| **Merges landing / Span** | Each round's merges as they land; rounds against merges (sequential BPE is the diagonal). |
| **Benchmarks** | `results/*.csv`: phase speedups, phase 3 per engine, owner load. |
| **Why a batch is exact / Hugging Face** | The theorem; our merges loaded into HF `tokenizers` give the reference's token ids. |

**Honest limits.** Core tiles and P/E badges are sampled when each event starts, and threads migrate: in phase 1 (17 ms
units) the sample says little, in phase 3 (microsecond events) it holds. Phase 2 and the heap updates are timed but not
traced per event. The word view shows 400 of ~600k distinct words. A run trained on battery is real but slower; report
numbers come from `bench/run_bench.py --require-ac`.

## Before the demo
On mains power, train a WikiText 32k "race set" (Race card), and optionally an `incr-batch` run with **batch every
round** ticked for the span figure. Studio shows the newest run, so train the race set last. URL stills: `#end`
(finished run), `#at=0.3` (30% through), `#p3` (phase 3 view), `&light` / `&dark`.

## Script (5–6 minutes; numbers measured on WikiText-103, vocab 32k)
1. **Race.** Play. "Left: the standard algorithm on one thread. Right: ours on 11 threads, merges on the 5 fast cores.
   Same data, each at its recorded speed." At the flag: "about 4× on the whole run (3.9× in the benchmarks), and both
   wrote the same `merges.txt`."
2. **Replay.** "Each row is a thread. Blue: all 11 read the text in parallel. Then the merge phase: the master picks a
   batch (yellow), the threads rewrite words together (green)." The badge turns green: identical to IncrSerial.
3. **The work.** "Each square is a megabyte of Wikipedia; threads take the next one when free, so the work evens out."
   Press Next round: "five threads rewrote these words in the same round; the underlined token is new."
4. **Span + theorem.** "Normal BPE does one merge per step. With every round batched: 31,743 merges in 2,561 rounds,
   up to 59 at once. A batch is the best pairs sharing no token, cut where a tie could reorder them; proved equal to
   sequential BPE and checked at runtime."
5. **Hugging Face.** Encode: same token ids as the reference encoder.
6. **Benchmarks.** "Pre-tokenizing: 4.0× at 5 threads, 5.1× at 11. The merge chain: IncrSerial stays at 1.0×, one
   merge per step gets 1.15×, batching 1.63×. On the E-cores it drops to 1.38×; keeping merges on the P-cores
   (`--merge-threads auto`) holds it at the 5-thread level."

**Don't mix modes:** 2,561 rounds is the every-round-batched run; the default (fast) run takes 19,888 rounds and gives
the 1.63×. **Don't claim** phase-1 work ran on specific core types (see the limits above).
