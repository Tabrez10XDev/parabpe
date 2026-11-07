Trace: 5 threads, 70522 events. Cores: cpu < 6 is an E-core (hw.perflevel1.logicalcpu; Apple numbers E-cores first).

| event | count | total ms | mean µs | E-core share |
|---|---:|---:|---:|---:|
| CHUNK | 6 | 739.4 | 123240.9 | 0% |
| PRETOK_UNIT | 508 | 5436.0 | 10700.9 | 1% |
| SHARD_REDUCE | 6872 | 147.3 | 21.4 | 0% |
| SELECT | 19888 | 71.8 | 3.6 | 0% |
| SERIAL_APPLY | 19030 | 66.0 | 3.5 | 0% |
| OWNER_MERGE | 4290 | 602.5 | 140.5 | 0% |
| ZERO_SWEEP | 40 | 19.8 | 494.3 | 0% |
| ORDER_UPDATE | 19888 | 39.4 | 2.0 | 0% |

Phase 3 busy time per thread (ms): t0 303.8, t1 123.6, t2 123.5, t3 124.9, t4 123.7
