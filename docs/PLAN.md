# ParaBPE: architecture and method

The design as built. The original plan (timeline, team split, build order) is in git history:
`git show 6b378a8:docs/PLAN.md`. The training rules themselves are in [SPEC.md](../SPEC.md).

## Goals
1. **Exactness.** Every parallel variant writes a `merges.txt` and `vocab.json` byte-identical to the serial baseline,
   at every thread count.
2. **Honest speedup.** Measured against the best serial trainer (IncrSerial), per phase, with strong and weak scaling
   and an Amdahl fit.
3. **The story.** Phase 1 (pre-tokenize) scales; phase 3 (the merge chain) is the Amdahl term. Batched rounds are how we
   attack it.

## Stack
C++20 + OpenMP · PCRE2 10.45 with JIT (Unicode 16) · `ankerl::unordered_dense` · CMake, GoogleTest, CLI11 ·
Python (`regex`, pandas, matplotlib, `tokenizers`) for the reference, benchmarks and plots · GitHub Actions: GCC, ASan,
and TSan (clang + LLVM libomp) · LaTeX (tectonic) for the report.

## Pipeline
```
corpus (UTF-8, <|endoftext|>-separated)  --mmap-->  chunker: special tokens -> documents -> ~1 MiB work units
PHASE 1  pre-tokenize: OpenMP, schedule(dynamic,1) over units; PCRE2 per thread; per-thread word maps
         pre-bucketed by the LOW hash bits; after a barrier, bucket b of every thread -> shard b (no locks)
         WordTable: struct of arrays (tokens, offset, len, freq)
PHASE 2  pair counts: owners (word sets) build private pair -> words indexes; sharded global counts; lazy max-heap
PHASE 3  merge engine (below), then the writers: merges.txt, vocab.json, timings.json
```

## Merge engines
| Engine | How | Role |
|---|---|---|
| `naive-serial` | Recount every pair for every merge | Oracle on small inputs |
| `naive-par` | The same recount, in parallel | Shows that recounting loses to incremental updates |
| `incr-serial` | Inverted index + whole-word diffs + lazy max-heap | **The speedup baseline** |
| `incr-par` | One merge per step; big steps (≥ 4096 tokens touched) split across owner threads | One-merge-per-step parallelism |
| `incr-batch` | Batched rounds of token-disjoint pairs, proved equal to sequential BPE; big rounds parallel over hashed owners | This project's contribution |

Details of the batch rule and its proof: [docs/notes/batched_merging.md](notes/batched_merging.md).
`--merge-threads auto` runs phase 3 on the performance cores while phase 1 uses every thread.

## Verification
1. **Golden files:** 11 small corpora with outputs from the independent Python reference (`reference/bpe_ref.py`).
2. **Equivalence matrix:** every pre-tokenizer × engine at threads {1,2,3,4,8,16} (and mixed merge teams) on 10 MB
   slices of both corpora must hash identically to the naive serial oracle.
3. **Runtime oracle:** `--verify-batches` checks every member of every round against a sequential engine.
4. **Interop:** our files load into Hugging Face `tokenizers` and encode held-out text to the reference's token ids.
5. **Sanitizers and mutations:** ASan and TSan in CI; every change to exactness logic broken on purpose to show a test
   catches it.

## Benchmark method
- Threads {1,2,3,4,5} (performance cores) reported apart from {6,8,11} (efficiency cores join); macOS cannot pin threads.
- One warm-up run, then 5 measured runs (3 for the largest inputs); medians with a min–max band; warm page cache.
- Mains power only (`--require-ac`); stalled runs (wall > 2 × CPU + 0.05 s) are dropped and counted.
- Every row records its `merges.txt` sha256, so the benchmark doubles as an exactness check.
- Plots: per-phase speedup, stacked phase times, Amdahl fit, phase-1 weak scaling, threshold sweep, span, owner load,
  merge team (`bench/plot.py`); numbers for the report from `report/numbers.py`.

## What changed from the original plan
- IncrPar forks only for big steps instead of one persistent region: most of the ~31k merges touch a few words.
- Batched merging (B0–B5) was added after profiling showed phase 3 bounding the speedup.
- ASan and TSan run only in CI (their runtimes hang on this macOS); profiling used Instruments, not `perf`.
- The C++ `encode` command was dropped: the Python reference encodes, and the Hugging Face check covers interop.
