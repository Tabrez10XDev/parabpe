# Profiling notes

## N8 spike (2026-11-03)

Produced by `scripts/profile_phases.py`; raw outputs are in `runs/n8/`. Machine: M3 Pro (5 P + 6 E cores),
macOS 27.0.1, **on battery**. Each row is one run, so these numbers come from a spike and do not follow the S8
benchmark method. Each "threads" figure is the number of OpenMP threads (or Rayon threads for HF).

### Python reference (`bpe_ref.py`) on the 10 MB slices, vocab 1000

| phase | TinyStories (10k unique words) | WikiText (71k unique words) |
|---|---:|---:|
| read + decode | 0.01 s | 0.01 s |
| split on specials | 0.01 s | 0.00 s |
| pre-tokenize + count words | 0.81 s (8%) | 0.68 s (1%) |
| first pair count (phase 2) | 0.01 s | 0.08 s |
| merge loop (743 merges, full recount each step) | **9.10 s (92%)** | **73.82 s (99%)** |
| total | 9.94 s | 74.59 s |

### HF `tokenizers` vs parabpe on 100 MB, vocab 32000

| run | TinyStories total | WikiText total | WikiText phases (chunk / pretok / build+pairs / merge) |
|---|---:|---:|---|
| HF, `RAYON_NUM_THREADS=1` | 15.20 s | 14.26 s | n/a |
| HF, `RAYON_NUM_THREADS=11` | 4.13 s | 4.33 s | n/a |
| parabpe serial / incr-serial, 1 thread | 0.99 s | 1.27 s | 0.02 / 0.96 / 0.02 / 0.27 |
| parabpe omp / incr-serial, 11 threads | 0.21 s | 0.52 s | 0.02 / 0.21 / 0.02 / 0.27 |

- WikiText gives 31,743 merges for both tools.
- On TinyStories the merging stops early: HF makes 28,035 merges and we make 28,057.
- HF is not doing identical work. It does not split on `<|endoftext|>`, so it pre-tokenizes the separator as text.
  It also includes its own file reading. Treat the comparison as ballpark only.

### parabpe phase breakdown on the largest inputs, vocab 32000

| input | threads | chunk | pretok | build + pairs | merge | total |
|---|---:|---:|---:|---:|---:|---:|
| `wikitext.full` (540 MB) | 1 (serial) | 1.06 s (cold) | 5.42 s | 0.04 s | 0.64 s | 7.17 s |
| `wikitext.full` | 11 (omp) | 0.10 s (warm) | 1.14 s | 0.04 s | 0.68 s | 1.98 s |
| `tinystories.1G` | 11 (omp), 1st run | 1.90 s (cold) | 1.52 s | 0.00 s | 0.05 s | 3.48 s |
| `tinystories.1G` | 11 (omp), 2nd run | 0.20 s (warm) | 1.41 s | 0.00 s | 0.05 s | 1.66 s |

### Takeaways
- **Serial time is mostly phase 1 in C++.** In the Python reference the naive merge loop takes 92–99% of the time,
  which is why IncrSerial is the speedup baseline. With IncrSerial in C++ the picture flips: pre-tokenization is
  76% (WikiText) to 95% (TinyStories) of serial time.
- **The merge loop limits the speedup on vocabulary-rich text.** Phase 1 scales well: 4.7× on `wikitext.full` and
  7.0× on `tinystories.1G` at 11 threads. After that, the serial merge loop is 34% of the 11-thread wall time on
  `wikitext.full`, but only 3% on TinyStories. S5/S6 can only pay off on WikiText-like corpora. G1 decides how
  much effort S6 gets.
- **Benchmark hygiene matters more than expected.**
  - Cold page cache inflates `chunk` by 10× on 1 GB (1.90 s vs 0.20 s), because the mmap faults land in the UTF-8
    check. The N7 harness must discard a warm-up run and report cold vs warm separately, or drop caches
    consistently.
  - This spike ran on battery. S8 must run on mains power.
