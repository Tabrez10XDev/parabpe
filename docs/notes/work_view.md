# The work view (Studio, B10): the data, not just the timeline

The **The work** card shows the same replay clock as the swimlanes, drawn on the data the run worked on. Code:
`studio/studio.js` (section "the work"), `src/trace_words.{hpp,cpp}`, the `unit` index on `PRETOK_UNIT` events
(`src/pretok_omp.cpp`). Train a run in Studio, press Play in the Replay card (or step round by round in this one).

## Phase 1: the corpus as work units
One cell per work unit (about 1 MB of whole documents), in corpus order. A cell is outlined while some thread is
pre-tokenizing it and filled when done. Under it, units finished per thread.

- **What it teaches:** `schedule(dynamic, 1)`. Each thread takes the next unit as soon as it finishes one, so the
  strip fills from the left in a band about as wide as the team, and the work evens out across threads on its own
  (wikitext.full, 11 threads: 45–47 units per thread).
- **Measured, and different from what I expected:** units that *started* on an E-core took about as long as those
  that started on a P-core (wikitext.full, 11 threads: median 15.7 ms against 17.3 ms; 34 % started on an E-core).
  A unit is long enough for macOS to move its thread between core types, so the core sampled at the start of a
  17 ms event says little about where it ran. That is why the strip is not coloured P/E, and why the card prints
  this measurement instead. (Phase-3 events last microseconds, so their samples are much more telling.)
- Data: `PRETOK_UNIT` events carry `"unit": i` (the work unit's index; B10). Older traces lack it and the card says
  so. The serial pre-tokenizer is timed, not traced per unit.

## Phase 3: real words being rewritten
- **The words.** `parabpe train --trace-words FILE` writes the 400 most frequent words of 4+ bytes (`" the"`,
  `" and"`, `" British"`…) with each word's index in the word table and its **owner**: the hashed owner IncrBatch
  gives it for phase 3 (`hashed_owner(w, batch_owner_count(merge_cfg))`, the same helpers the engine uses). It is
  written after the run's clock stops and changes nothing (test: merges identical with it on).
- **Their tokens.** Studio replays the run's `merges.txt` on each word's bytes: merge *i* rewrites every adjacent
  `(a, b)` left to right, which is what training does to every word. Checked: for all 400 words on wikitext.10M v4000
  and wikitext.full v32k, the final tokens equal the Python reference's rank-based encoder (`bpe_ref.encode`, a
  different algorithm). Within a round the order of its merges doesn't matter: they share no token.
- **Who rewrote them.** A round is selected (`SELECT`), its words are rewritten, then the order is updated
  (`ORDER_UPDATE`). Between the two, a parallel round has one `OWNER_MERGE` event per owner, with the thread and CPU
  that ran it; a word's thread is the one that ran its owner. A small round has a `SERIAL_APPLY` on the master
  instead, and the card says "the master rewrote every word alone". Nothing about the assignment is guessed.
- **What you see.** "This round, by thread": one row per thread, each with the sampled words it rewrote (before →
  after, the token the round created underlined). The wall: the 60 most frequent words as they stand now, this
  round's words outlined with their thread. Over phase 3 the wall goes from single bytes to whole-word tokens.
  The step buttons jump between rounds that change a word on the list.
- **With `--merge-threads auto`** the lanes are t0–t4, all on P-cores; without it, up to 11 threads, some on
  E-cores, and each round waits for the slowest.

## What it does not show
- Every word: 400 of 598,983 distinct words (wikitext.full). The rest are rewritten the same way.
- Phase 2 (word table, pair counts) and the heap updates in `ORDER_UPDATE`: timed, but no words to show.
- Owners for IncrPar or the serial engines: only IncrBatch's owners are recorded, and only its rounds are traced.
