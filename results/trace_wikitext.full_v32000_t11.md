# Trace summary: wikitext.full, vocab 32000, 11 threads, omp, incr-batch (default mode), mains power; Studio run 798749d713a2; merges.txt identical to IncrSerial.

Trace: 11 threads, 82596 events. Cores: cpu < 6 is an E-core (hw.perflevel1.logicalcpu; Apple numbers E-cores first).

| event | count | total ms | mean µs | E-core share |
|---|---:|---:|---:|---:|
| CHUNK | 12 | 469.3 | 39110.3 | 7% |
| PRETOK_UNIT | 508 | 8996.6 | 17709.9 | 37% |
| SHARD_REDUCE | 13744 | 261.3 | 19.0 | 38% |
| SELECT | 19888 | 79.0 | 4.0 | 0% |
| SERIAL_APPLY | 19030 | 78.8 | 4.1 | 0% |
| OWNER_MERGE | 9438 | 782.7 | 82.9 | 54% |
| ZERO_SWEEP | 88 | 41.9 | 476.6 | 65% |
| ORDER_UPDATE | 19888 | 41.0 | 2.1 | 0% |

Phase 3 busy time per thread (ms): t0 276.9, t1 90.8, t2 74.7, t3 76.6, t4 70.7, t5 72.3, t6 69.8, t7 71.5, t8 73.2, t9 71.7, t10 75.2

Phase 3 E-core share per thread: t0 0%, t1 100%, t2 1%, t3 0%, t4 0%, t5 100%, t6 100%, t7 100%, t8 0%, t9 100%, t10 100%
