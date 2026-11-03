#!/usr/bin/env python3
"""Benchmark runner (N7): runs a YAML grid of parabpe configurations and appends one CSV row per
run to results/<config name>.csv. Each configuration gets one warm-up run first (warm page
cache, PLAN.md §6), stored with warmup=1 and ignored by plot.py.

    .venv/bin/python -I bench/run_bench.py bench/configs/smoke.yaml

YAML keys (lists unless noted): inputs, threads, pretok, merge, vocab_size, reps (int, default 5),
pretok_reduce (default [bucketed]), par_threshold_tokens (default [4096]), zip (bool: pair
inputs[i] with threads[i] instead of the full product, for weak scaling).
Configurations with only serial strategies run at threads=1 only. On macOS the runner holds a
`caffeinate` assertion (a system sleep mid-run once turned 0.1 s into 295 s) and records each
run's CPU time (`cpu_s`), so stalls show up as total >> cpu_s / threads. Strategies that report
"not implemented" are skipped with a note.
"""

import argparse
import csv
import datetime
import hashlib
import itertools
import json
import os
import pathlib
import resource
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
PHASES = ["chunk", "pretok", "build", "pairs", "merge", "total"]  # CSV columns <phase>_s
COLUMNS = (["timestamp", "input", "input_bytes", "vocab_size", "threads", "pretok", "pretok_reduce",
            "merge", "par_threshold_tokens", "work_unit_bytes", "rep", "warmup", "merges"] +
           [f"{p}_s" for p in PHASES] +
           ["cpu_s", "merges_sha256", "power", "cpu_model", "physical_cores", "performance_cores",
            "efficiency_cores", "os", "git_sha"])
PARALLEL = {"pretok": {"omp"}, "merge": {"naive-par", "incr-par"}}


def as_list(v):
    return v if isinstance(v, list) else [v]


def grid(cfg):
    inputs, threads = as_list(cfg["inputs"]), as_list(cfg["threads"])
    pairs = list(zip(inputs, threads)) if cfg.get("zip") else list(itertools.product(inputs, threads))
    seen = set()
    for (inp, t), vocab, pretok, merge, reduce, thr in itertools.product(
            pairs, as_list(cfg["vocab_size"]), as_list(cfg["pretok"]), as_list(cfg["merge"]),
            as_list(cfg.get("pretok_reduce", "bucketed")),
            as_list(cfg.get("par_threshold_tokens", 4096))):
        parallel = pretok in PARALLEL["pretok"] or merge in PARALLEL["merge"]
        if not parallel:
            t = 1  # a serial-only configuration does not depend on the thread count
        if pretok != "omp":
            reduce = "bucketed"
        if merge != "incr-par":
            thr = 4096
        key = (inp, t, vocab, pretok, merge, reduce, thr)
        if key not in seen:
            seen.add(key)
            yield dict(input=inp, threads=t, vocab_size=vocab, pretok=pretok, merge=merge,
                       pretok_reduce=reduce, par_threshold_tokens=thr)


def power_source():
    if sys.platform != "darwin":
        return "n/a"
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        return out.splitlines()[0].split("'")[1]
    except (OSError, IndexError):
        return "unknown"


def launcher():
    """numactl prefix on multi-socket Linux machines (PLAN.md §6); empty elsewhere."""
    if not shutil.which("numactl"):
        return []
    out = subprocess.run(["numactl", "--hardware"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("available:") and int(line.split()[1]) > 1:
            return ["numactl", "--cpunodebind=0", "--membind=0"]
    return []


def run_one(binary, prefix, run, out_dir):
    cmd = prefix + [binary, "train", "--input", run["input"], "--vocab-size", str(run["vocab_size"]),
                    "--threads", str(run["threads"]), "--pretok", run["pretok"],
                    "--pretok-reduce", run["pretok_reduce"], "--merge", run["merge"],
                    "--par-threshold-tokens", str(run["par_threshold_tokens"]), "--out", str(out_dir)]
    env = dict(os.environ, OMP_PROC_BIND="close", OMP_PLACES="cores")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if proc.returncode != 0:
        return None, proc.stderr.strip()
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    t = json.loads((out_dir / "timings.json").read_text())
    sha = hashlib.sha256((out_dir / "merges.txt").read_bytes()).hexdigest()
    return (t, sha, cpu), None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--binary", default=str(ROOT / "build" / "parabpe"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--reps", type=int, help="override the config's reps")
    ap.add_argument("--require-ac", action="store_true", help="refuse to run on battery (macOS)")
    ap.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    args = ap.parse_args()

    cfg_path = pathlib.Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    reps = args.reps or cfg.get("reps", 5)
    runs = list(grid(cfg))
    power = power_source()
    print(f"{cfg_path.stem}: {len(runs)} configurations x (1 warm-up + {reps} reps), power: {power}")
    if args.dry_run:
        for r in runs:
            print("  ", r)
        return 0
    if power not in ("AC Power", "n/a"):
        msg = f"not on mains power ({power}); results will be marked"
        if args.require_ac:
            print(f"error: {msg}")
            return 1
        print(f"warning: {msg}")
    for inp in {r["input"] for r in runs}:
        if not pathlib.Path(inp).is_file():
            print(f"error: missing input {inp}")
            return 1

    out_csv = pathlib.Path(args.results_dir) / f"{cfg_path.stem}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        with out_csv.open(newline="") as f:
            header = next(csv.reader(f), None)
        if header != COLUMNS:
            print(f"error: {out_csv} has a different header; move it away first")
            return 1
    new_file = not out_csv.exists()
    prefix = launcher()
    if shutil.which("caffeinate"):  # no idle/system sleep while this process lives
        subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
    shas = {}  # (input, vocab_size) -> merges.txt sha256: every strategy must agree
    failures = 0
    with out_csv.open("a", newline="") as f, tempfile.TemporaryDirectory(prefix="bench_") as tmp:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        for i, run in enumerate(runs, 1):
            label = (f"[{i}/{len(runs)}] {pathlib.Path(run['input']).name} v{run['vocab_size']} "
                     f"{run['pretok']}/{run['pretok_reduce']}/{run['merge']} t{run['threads']}")
            for rep in range(reps + 1):
                result, err = run_one(args.binary, prefix, run, pathlib.Path(tmp))
                if err:
                    note = "skipped (not implemented)" if "not implemented" in err else f"FAILED: {err}"
                    failures += "not implemented" not in err
                    print(f"{label}: {note}")
                    break
                t, sha, cpu = result
                key = (run["input"], run["vocab_size"])
                if shas.setdefault(key, sha) != sha:
                    print(f"{label}: MISMATCH merges.txt differs from the other strategies")
                    failures += 1
                hw = t["hardware"]
                w.writerow({
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    **{k: run[k] for k in ("input", "vocab_size", "threads", "pretok", "pretok_reduce",
                                           "merge", "par_threshold_tokens")},
                    "input_bytes": pathlib.Path(run["input"]).stat().st_size,
                    "work_unit_bytes": t["config"]["work_unit_bytes"],
                    "rep": rep, "warmup": int(rep == 0), "merges": t["merges"],
                    **{f"{p}_s": t["timings"][p] for p in PHASES},
                    "cpu_s": round(cpu, 6), "merges_sha256": sha, "power": power,
                    **{k: hw.get(k, "") for k in ("cpu_model", "physical_cores", "performance_cores",
                                                  "efficiency_cores", "os")},
                    "git_sha": t["git_sha"],
                })
                f.flush()
                if rep == reps:
                    print(f"{label}: last total {t['timings']['total']:.3f}s")
    print(f"wrote {out_csv}; {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
