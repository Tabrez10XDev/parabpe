# Trace overhead (B6)

wikitext.full, 5 threads, omp pre-tokenizer, mains power, median of 7 interleaved runs (min-max for phase 3).
pre = binary before B6 (d80bce1); new = B6 binary, trace off; on = B6 binary with --trace. Seconds.

```
pre incr-serial v32000     pretok 1.250  merge 0.6006 [0.5912-0.6189]  total 1.919
new incr-serial v32000     pretok 1.250  merge 0.6008 [0.5971-0.6347]  total 1.950
pre incr-batch v32000      pretok 1.249  merge 0.3593 [0.3541-0.3764]  total 1.673
new incr-batch v32000      pretok 1.243  merge 0.3601 [0.3540-0.3746]  total 1.670
on  incr-batch v32000      pretok 1.237  merge 0.3602 [0.3545-0.3678]  total 1.657
pre incr-serial v100000    pretok 1.254  merge 0.8905 [0.8881-0.9075]  total 2.222
new incr-serial v100000    pretok 1.233  merge 0.8869 [0.8707-0.8974]  total 2.201
pre incr-batch v100000     pretok 1.254  merge 0.6423 [0.6259-0.6698]  total 1.958
new incr-batch v100000     pretok 1.242  merge 0.6453 [0.6237-0.6583]  total 1.945
on  incr-batch v100000     pretok 1.243  merge 0.6529 [0.6327-0.6657]  total 1.965
```
