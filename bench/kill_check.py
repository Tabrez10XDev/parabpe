#!/usr/bin/env python3
"""B4 kill check: prints phase 3 speedups from results/b4_sweep.csv and the verdict.

    .venv/bin/python -I bench/kill_check.py results/b4_sweep.csv

One default has to serve both vocabularies, so the candidate is the (owners, threshold) pair with
the best geometric mean speedup at 5 threads over vocab 32k and 100k. The check fails only if that
pair is below 1.3x at both. Exit code: 0 pass, 1 kill, 2 cannot decide (not mains power, missing
runs, or merges.txt differing from IncrSerial).
"""

import argparse
import math

import pandas as pd

THREADS, LIMIT = 5, 1.3
VOCABS = [32000, 100000]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df = df[df["warmup"] == 0]

    problems = []
    if not set(df["power"]) <= {"AC Power", "n/a"}:
        problems.append(f"runs not on mains power: {sorted(set(df['power']))}")
    if df.groupby("vocab_size")["merges_sha256"].nunique().max() > 1:
        problems.append("merges.txt differs between strategies")

    batch = df[df["merge"] == "incr-batch"]
    table = {}
    for (t, v), g in batch.groupby(["threads", "vocab_size"]):
        serial = df[(df["merge"] == "incr-serial") & (df["vocab_size"] == v)]["merge_s"].median()
        med = g.groupby(["owners_per_thread", "par_threshold_tokens"])["merge_s"].median()
        speedup = (serial / med).unstack("par_threshold_tokens")
        table[(t, v)] = speedup
        print(f"\nthreads {t}, vocab {v}: IncrSerial phase 3 {serial:.3f} s")
        print("speedup, owners per thread (rows) x threshold (columns)")
        print(speedup.round(2).to_string())

    if not all((THREADS, v) in table for v in VOCABS):
        problems.append(f"need {THREADS}-thread runs at vocab {VOCABS}")
    else:
        a, b = (table[(THREADS, v)] for v in VOCABS)
        geo = (a * b).map(math.sqrt)
        owners, threshold = geo.stack().idxmax()
        x, y = a.loc[owners, threshold], b.loc[owners, threshold]
        print(f"\ncandidate default: owners_per_thread={owners}, threshold={threshold}")
        print(f"phase 3 at {THREADS} threads: {x:.2f}x at v32k, {y:.2f}x at v100k")
        verdict = "KILL: below" if x < LIMIT and y < LIMIT else "PASS: not below"
    for p in problems:
        print(f"CANNOT DECIDE: {p}")
    if problems:
        return 2
    print(f"{verdict} {LIMIT}x at both vocabularies")
    return 1 if verdict.startswith("KILL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
