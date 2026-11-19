# ParaBPE task board

Every task is done except N5 (optional). The original cards, with acceptance criteria and files to touch, are in git
history: `git show 6b378a8:docs/TASKS.md`. Each finished task was committed on its own, so `git log --grep "^B4"`
finds its work.

| Task | What | Status / outcome |
|---|---|---|
| N1 | Repo skeleton, CMake, CI, smoke tests | ✅ |
| N2 | Spec + interfaces | ✅ draft (team review never held) |
| N3 | Data scripts (`get_data.sh`, `slice_data.py`) | ✅ TinyStories + WikiText-103; enwik9 path written, not run |
| N4 | Python reference + golden files | ✅ 11 golden cases, Unicode 16 pinned |
| S1 | Chunker + serial pre-tokenizer | ✅ golden `words.tsv` byte-exact |
| S2 | WordTable + NaiveSerial + writers + `train()` | ✅ byte-exact `merges.txt` / `vocab.json` |
| S3 | IncrSerial + equivalence harness | ✅ 125–266× faster merge phase than NaiveSerial |
| S4 | OMP pre-tokenizer + TSan CI | ✅ 7.0× at 11 threads (1 GB, one run); TSan green in CI |
| S5 | Parallel pair statistics | ✅ |
| S6 | IncrPar | ✅ 1.15× phase 3 at 5 threads (fork per big step, not a persistent region) |
| S7 | NaivePar | ✅ |
| S10 | Parallel chunker | ✅ |
| N7 | Bench harness | ✅ `bench/run_bench.py`, `bench/plot.py` |
| N8, G1 | Profiling spike and gate | ✅ phase 1 is 76–95% of serial time; modest S6 + S10 chosen |
| B0 | Batched merging: rule, proof, `tools/batchsim` | ✅ `docs/notes/batched_merging.md` |
| B1–B3 | Refactor, `incr-batch` plumbing, serial batched engine + `--verify-batches` | ✅ |
| B4 | Hashed owners + parallel rounds (kill check) | ✅ passed: 1.83× / 1.39× phase 3 at 5 threads |
| B5 | IncrPar owner-map ablation | ✅ load 2.30 → 1.08, but little speed gain alone |
| N6 | Related work | ✅ `docs/notes/related_work.md` |
| S8 | Full benchmarks (mains power) | ✅ `results/`: incr-batch 1.63× phase 3 at 5 threads, 3.91× whole run at 11 |
| B6 | `--trace` (Chrome Trace JSON, P/E samples) | ✅ |
| B7 | ParaBPE Studio | ✅ `studio/` |
| S9 | Report | ✅ `report/report.pdf` |
| B8 | `--merge-threads N\|auto` (phase 3 on the P-cores) | ✅ 1.70× at 11 threads with auto, against 1.47× without |
| B9 | Studio: the race | ✅ |
| B10 | Studio: the work view | ✅ |
| N5 | Big-machine access | ⬜ optional, human task |
