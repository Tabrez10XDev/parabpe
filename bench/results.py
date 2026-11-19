"""Loading rules for results/*.csv, shared by bench/plot.py, bench/kill_check.py and
report/numbers.py, so every figure and every number in the report filters the same rows the same
way. Also the mains-power check used by run_bench.py and scripts/profile_phases.py.

- Warm-up rows (warmup = 1) are dropped.
- Stalled rows are dropped: wall time > 2 x CPU time + 0.05 s (the machine slept or was busy).
- Only empty cells are missing values; "n/a" (power off macOS) stays a string.
- name = the input file's stem (data/wikitext.full.txt -> wikitext.full).
- phase1_s = chunk_s + pretok_s: phase 1 as the report and Studio define it.
"""

import pathlib
import subprocess
import sys

import pandas as pd


def power_source():
    """'AC Power' or 'Battery Power' from pmset on macOS; 'n/a' elsewhere, 'unknown' if unreadable."""
    if sys.platform != "darwin":
        return "n/a"
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        return out.splitlines()[0].split("'")[1]
    except (OSError, IndexError):
        return "unknown"


def read(paths):
    """The rows of one or more CSVs, warm-ups included."""
    return pd.concat([pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in paths],
                     ignore_index=True)


def stalled(df):
    return df["total_s"] > 2 * df["cpu_s"] + 0.05


def clean(df):
    """(rows without warm-ups and stalls, with name and phase1_s added; number of stalls dropped)."""
    df = df[df["warmup"] == 0].copy()
    bad = stalled(df)
    df = df[~bad].copy()
    df["name"] = df["input"].map(lambda p: pathlib.Path(p).stem)
    df["phase1_s"] = df["chunk_s"] + df["pretok_s"]
    return df, int(bad.sum())


def load(*paths):
    """read() then clean(), without the stall count."""
    return clean(read(paths))[0]
