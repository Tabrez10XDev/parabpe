Trace: 5 threads, 273975 events. Cores: cpu < 6 is an E-core (hw.perflevel1.logicalcpu; Apple numbers E-cores first).

| event | count | total ms | mean µs | E-core share |
|---|---:|---:|---:|---:|
| CHUNK | 6 | 40.7 | 6784.3 | 0% |
| PRETOK_UNIT | 508 | 5454.0 | 10736.2 | 1% |
| SHARD_REDUCE | 7624 | 148.3 | 19.5 | 0% |
| SELECT | 87313 | 194.1 | 2.2 | 0% |
| SERIAL_APPLY | 86361 | 160.3 | 1.9 | 0% |
| OWNER_MERGE | 4760 | 606.3 | 127.4 | 3% |
| ZERO_SWEEP | 90 | 43.3 | 480.7 | 0% |
| ORDER_UPDATE | 87313 | 73.3 | 0.8 | 0% |

Phase 3 busy time per thread (ms): t0 559.4, t1 130.7, t2 128.5, t3 130.6, t4 128.0
