#!/usr/bin/env python3
"""Profiling spike (N8): where does BPE training time go?

1. bpe_ref.py phases (read, split, pre-tokenize, first pair count, merges) on a small slice.
2. HF `tokenizers` BPE training at RAYON_NUM_THREADS=1 and =N on a larger slice, next to our C++
   binary (serial/incr-serial at 1 thread, omp/incr-serial at N threads) on the same input.

Prints a Markdown report (also written to --out). Run with `python -I` from the repo root:

    .venv/bin/python -I scripts/profile_phases.py --small data/tinystories.10M.txt \\
        --large data/tinystories.100M.txt --threads 11 --out runs/n8/profile.md
"""

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPECIAL = "<|endoftext|>"


def ref_phases(path, vocab_size):
    sys.path.insert(0, str(ROOT / "reference"))
    import bpe_ref

    t = {}
    t0 = time.perf_counter()
    text = pathlib.Path(path).read_bytes().decode("utf-8")
    t1 = time.perf_counter()
    docs = bpe_ref.split_docs(text, [SPECIAL])
    t2 = time.perf_counter()
    words = bpe_ref.pretokenize(docs)
    t3 = time.perf_counter()
    bpe_ref.count_pairs({tuple(w): f for w, f in words.items()})
    t4 = time.perf_counter()
    merges, _ = bpe_ref.train(words, vocab_size, [SPECIAL])
    t5 = time.perf_counter()
    t["read + decode"] = t1 - t0
    t["split on specials"] = t2 - t1
    t["pre-tokenize + count words"] = t3 - t2
    t["first pair count (phase 2)"] = t4 - t3
    t[f"merge loop ({len(merges)} merges, recounts every step)"] = t5 - t4
    return t, len(words)


def hf_worker(path, vocab_size):
    from tokenizers import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer(add_prefix_space=False)
    t0 = time.perf_counter()
    tok.train([path], vocab_size=vocab_size, min_frequency=1, special_tokens=[SPECIAL],
              show_progress=False)
    print(json.dumps({"seconds": time.perf_counter() - t0, "vocab": tok.get_vocab_size()}))


def hf_train(path, vocab_size, threads):
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    cmd = [sys.executable, "-I", __file__, "--hf-worker", path, "--vocab-size", str(vocab_size)]
    out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def cpp_train(binary, path, vocab_size, threads, pretok):
    with tempfile.TemporaryDirectory(prefix="parabpe_n8_") as tmp:
        subprocess.run([binary, "train", "--input", path, "--vocab-size", str(vocab_size),
                        "--threads", str(threads), "--pretok", pretok, "--merge", "incr-serial",
                        "--out", tmp], check=True, capture_output=True)
        return json.loads((pathlib.Path(tmp) / "timings.json").read_text())


def power_source():
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        return out.splitlines()[0].split("'")[1]
    except (OSError, IndexError):
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--small", default="data/tinystories.10M.txt")
    ap.add_argument("--large", default="data/tinystories.100M.txt")
    ap.add_argument("--vocab-size", type=int, default=1000, help="bpe_ref.py run on --small")
    ap.add_argument("--hf-vocab-size", type=int, default=32000, help="HF vs C++ run on --large")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--binary", default=str(ROOT / "build" / "parabpe"))
    ap.add_argument("--out")
    ap.add_argument("--hf-worker", metavar="PATH", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.hf_worker:
        hf_worker(args.hf_worker, args.vocab_size)
        return

    lines = [f"Machine: {platform.machine()} {platform.mac_ver()[0] or platform.platform()}, "
             f"{os.cpu_count()} logical CPUs, power: {power_source()}. One run per row (a spike, "
             "not the S8 methodology).", ""]

    ref, n_words = ref_phases(args.small, args.vocab_size)
    total = sum(ref.values())
    lines += [f"### bpe_ref.py on `{args.small}`, vocab {args.vocab_size} ({n_words} unique words)",
              "", "| phase | seconds | share |", "|---|---:|---:|"]
    lines += [f"| {k} | {v:.2f} | {100 * v / total:.1f}% |" for k, v in ref.items()]
    lines += [f"| **total** | **{total:.2f}** | |", ""]

    n = args.threads
    rows = []
    for t in (1, n):
        hf = hf_train(args.large, args.hf_vocab_size, t)
        rows.append((f"HF tokenizers, RAYON_NUM_THREADS={t}", hf["seconds"],
                     hf["vocab"] - 256 - 1, None))
    for t, pretok in ((1, "serial"), (n, "omp")):
        c = cpp_train(args.binary, args.large, args.hf_vocab_size, t, pretok)
        rows.append((f"parabpe {pretok}/incr-serial, {t} thread(s)", c["timings"]["total"],
                     c["merges"], c["timings"]))
    lines += [f"### Training on `{args.large}`, vocab {args.hf_vocab_size}", "",
              "| run | merges | total (s) | chunk | pretok | build + pairs | merge |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, n_merges, secs, c in [(r[0], r[2], r[1], r[3]) for r in rows]:
        if c is None:
            lines.append(f"| {name} | {n_merges} | {secs:.2f} | | | | |")
        else:
            lines.append(f"| {name} | {n_merges} | {secs:.2f} | {c['chunk']:.2f} | {c['pretok']:.2f} | "
                         f"{c['build'] + c['pairs']:.3f} | {c['merge']:.2f} |")
    lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(report)


if __name__ == "__main__":
    main()
