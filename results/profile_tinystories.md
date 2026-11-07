Machine: arm64 27.0.1, 11 logical CPUs, power: AC Power. One run per row (a spike, not the S8 methodology).

### bpe_ref.py on `data/tinystories.10M.txt`, vocab 1000 (10074 unique words)

| phase | seconds | share |
|---|---:|---:|
| read + decode | 0.00 | 0.1% |
| split on specials | 0.01 | 0.1% |
| pre-tokenize + count words | 0.79 | 8.3% |
| first pair count (phase 2) | 0.01 | 0.1% |
| merge loop (743 merges, recounts every step) | 8.77 | 91.5% |
| **total** | **9.59** | |

### Training on `data/tinystories.100M.txt`, vocab 32000

| run | merges | total (s) | chunk | pretok | build + pairs | merge |
|---|---:|---:|---:|---:|---:|---:|
| HF tokenizers, RAYON_NUM_THREADS=1 | 28035 | 14.49 | | | | |
| HF tokenizers, RAYON_NUM_THREADS=11 | 28035 | 4.00 | | | | |
| parabpe serial/incr-serial, 1 thread(s) | 28057 | 0.95 | 0.01 | 0.90 | 0.001 | 0.03 |
| parabpe omp/incr-serial, 11 thread(s) | 28057 | 0.18 | 0.00 | 0.14 | 0.001 | 0.03 |
