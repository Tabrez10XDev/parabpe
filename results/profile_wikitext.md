Machine: arm64 27.0.1, 11 logical CPUs, power: AC Power. One run per row (a spike, not the S8 methodology).

### bpe_ref.py on `data/wikitext.10M.txt`, vocab 1000 (71249 unique words)

| phase | seconds | share |
|---|---:|---:|
| read + decode | 0.01 | 0.0% |
| split on specials | 0.00 | 0.0% |
| pre-tokenize + count words | 0.66 | 0.9% |
| first pair count (phase 2) | 0.08 | 0.1% |
| merge loop (743 merges, recounts every step) | 70.11 | 98.9% |
| **total** | **70.86** | |

### Training on `data/wikitext.100M.txt`, vocab 32000

| run | merges | total (s) | chunk | pretok | build + pairs | merge |
|---|---:|---:|---:|---:|---:|---:|
| HF tokenizers, RAYON_NUM_THREADS=1 | 31743 | 13.51 | | | | |
| HF tokenizers, RAYON_NUM_THREADS=11 | 31743 | 4.08 | | | | |
| parabpe serial/incr-serial, 1 thread(s) | 31743 | 1.17 | 0.01 | 0.89 | 0.015 | 0.24 |
| parabpe omp/incr-serial, 11 thread(s) | 31743 | 0.46 | 0.00 | 0.19 | 0.015 | 0.26 |
