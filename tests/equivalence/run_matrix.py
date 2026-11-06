#!/usr/bin/env python3
"""Equivalence matrix: every implemented (pretok, merge) pair x thread count must produce a
merges.txt (and vocab.json) byte-identical to the oracle, serial pre-tokenizer + NaiveSerial at
1 thread. Strategies that still report "not implemented" are listed as skipped.

    python3 tests/equivalence/run_matrix.py --input data/tinystories.10M.txt --vocab-size 4000

Exit codes: 0 all identical, 1 any mismatch or crash, 77 input missing (CTest skip).
"""

import argparse
import hashlib
import pathlib
import subprocess
import sys
import tempfile

PRETOKS = ["serial", "omp"]
MERGES = ["naive-serial", "incr-serial", "naive-par", "incr-par", "incr-par:hashed", "incr-batch"]
THREADS = [1, 2, 3, 4, 8, 16]
OUTPUTS = ["merges.txt", "vocab.json"]
SKIP = 77


def digest(out_dir):
    return tuple(hashlib.sha256((out_dir / f).read_bytes()).hexdigest() for f in OUTPUTS)


def train(binary, args, pretok, merge, threads, out_dir):
    engine, _, owner_map = merge.partition(":")
    cmd = [binary, "train", "--input", args.input, "--vocab-size", str(args.vocab_size),
           "--threads", str(threads), "--pretok", pretok, "--merge", engine, "--out", str(out_dir)]
    if owner_map:
        cmd += ["--owner-map", owner_map]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if "not implemented" in proc.stderr:
            return "skip", proc.stderr.strip()
        return "error", proc.stderr.strip()
    return "ok", digest(out_dir)


def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True)
    p.add_argument("--vocab-size", type=int, default=4000)
    p.add_argument("--binary", default=str(root / "build" / "parabpe"))
    p.add_argument("--threads", type=int, nargs="+", default=THREADS)
    args = p.parse_args()

    if not pathlib.Path(args.input).is_file():
        print(f"skip: {args.input} not found (run scripts/get_data.sh + slice_data.py)")
        return SKIP
    if 3 not in args.threads:
        print("warning: threads=3 is required for the equivalence check (CLAUDE.md)")

    failures, skipped = [], set()
    with tempfile.TemporaryDirectory(prefix="parabpe_matrix_") as tmp:
        tmp = pathlib.Path(tmp)
        status, oracle = train(args.binary, args, "serial", "naive-serial", 1, tmp / "oracle")
        if status != "ok":
            print(f"oracle failed: {oracle}")
            return 1
        print(f"oracle serial/naive-serial/t1  merges.txt sha256 {oracle[0][:16]}")
        for pretok in PRETOKS:
            for merge in MERGES:
                for t in args.threads:
                    if (pretok, merge) in skipped:
                        break
                    name = f"{pretok}/{merge}/t{t}"
                    status, result = train(args.binary, args, pretok, merge, t, tmp / name)
                    if status == "skip":
                        skipped.add((pretok, merge))
                        print(f"SKIP  {pretok}/{merge}: {result}")
                    elif status == "error":
                        failures.append(name)
                        print(f"ERROR {name}: {result}")
                    elif result != oracle:
                        failures.append(name)
                        bad = [f for f, a, b in zip(OUTPUTS, result, oracle) if a != b]
                        print(f"DIFF  {name}: {', '.join(bad)}")
                    else:
                        print(f"OK    {name}")

    print(f"\n{len(failures)} mismatches, {len(skipped)} strategy pairs not implemented yet")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
