#!/usr/bin/env python3
"""Plots and summary tables from bench/run_bench.py CSVs (N7).

    .venv/bin/python -I bench/plot.py results/*.csv --out-dir results/plots

Writes, per (input, vocab_size) present in the data:
  speedup_<input>_v<vocab>.png  per-phase speedup vs IncrSerial (serial pre-tokenizer, 1 thread):
                                median line, min/max band, ideal line, P-core / E-core regions
  phases_<input>_v<vocab>.png   stacked median phase times per thread count, one panel per strategy
plus amdahl.png (serial fraction fitted on threads 1-5), weak_scaling.png (phase 1, when inputs
grow with threads), threshold.png (IncrPar threshold sweep, when present) and summary.md
(median / min / max of every configuration, the table view of every plot).
Warm-up runs are ignored. Runs whose wall time is more than twice their CPU time are dropped as
stalls (e.g. the machine slept) and counted in summary.md.
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reference palette (light mode), fixed slot order; colour follows the entity, never its rank.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STRATEGY_COLORS = {
    "omp/incr-serial": SLOTS[0], "serial/incr-par": SLOTS[1], "omp/incr-par": SLOTS[2],
    "omp/incr-serial (critical)": SLOTS[3], "serial/naive-par": SLOTS[4],
    "serial/naive-serial": SLOTS[5], "omp/naive-par": SLOTS[6],
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
PHASES = ["chunk", "pretok", "build", "pairs", "merge"]
PHASE_LABELS = {"chunk": "chunk", "pretok": "phase 1: pre-tokenize", "build": "build",
                "pairs": "phase 2: pair stats", "merge": "phase 3: merges", "total": "total"}
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
P_CORES = 5  # M3 Pro: threads 1-5 fit on performance cores


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
        "ytick.color": TEXT_2, "text.color": TEXT, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 1, "grid.linestyle": "-", "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "lines.linewidth": 2,
        "legend.frameon": False, "axes.axisbelow": True,
    })


def strategy(row):
    s = f"{row['pretok']}/{row['merge']}"
    return s + " (critical)" if row["pretok"] == "omp" and row["pretok_reduce"] == "critical" else s


def load(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df["warmup"] == 0].copy()
    stalled = df["total_s"] > 2 * df["cpu_s"] + 0.05
    df["strategy"] = df.apply(strategy, axis=1)
    df["name"] = df["input"].map(lambda p: pathlib.Path(p).stem)
    return df[~stalled].copy(), int(stalled.sum())


def stats(df, keys):
    cols = [f"{p}_s" for p in PHASES + ["total"]]
    g = df.groupby(keys)[cols]
    return pd.concat([g.median(), g.min().add_suffix("_min"), g.max().add_suffix("_max")],
                     axis=1).reset_index()


def baseline(st, name, vocab):
    b = st[(st["name"] == name) & (st["vocab_size"] == vocab) &
           (st["strategy"] == "serial/incr-serial") & (st["threads"] == 1)
           & (st["par_threshold_tokens"] == 4096)]
    return None if b.empty else b.iloc[0]


def mark_regions(ax, max_t):
    if max_t > P_CORES:
        ax.axvspan(P_CORES + 0.5, max_t + 0.5, color=GRID, alpha=0.35, lw=0, zorder=0)
        ax.text(P_CORES + 0.7, 0.98, "E-cores join", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color=TEXT_2)


def speedup_plot(st, name, vocab, base, out):
    sub = st[(st["name"] == name) & (st["vocab_size"] == vocab) &
             (st["strategy"] != "serial/incr-serial") & (st["par_threshold_tokens"] == 4096)]
    if sub.empty:
        return False
    metrics = ["pretok", "merge", "total"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    max_t = int(sub["threads"].max())
    strategies = sorted(sub["strategy"].unique(), key=list(STRATEGY_COLORS).index)
    for ax, m in zip(axes, metrics):
        ax.plot([1, max_t], [1, max_t], color=TEXT_2, lw=1, label="ideal")
        mark_regions(ax, max_t)
        for i, s in enumerate(strategies):
            d = sub[sub["strategy"] == s].sort_values("threads")
            b = base[f"{m}_s"]
            color = STRATEGY_COLORS.get(s, SLOTS[-1])
            ax.fill_between(d["threads"], b / d[f"{m}_s_max"], b / d[f"{m}_s_min"], color=color,
                            alpha=0.18, lw=0)
            ax.plot(d["threads"], b / d[f"{m}_s"], color=color, marker=MARKERS[i], ms=8,
                    mec=SURFACE, mew=2, label=s)
            last = d.iloc[-1]
            ax.annotate(f"{b / last[f'{m}_s']:.1f}×", (last["threads"], b / last[f"{m}_s"]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=8,
                        color=TEXT)
        ax.set_title(f"{PHASE_LABELS[m]} (IncrSerial: {base[f'{m}_s']:.2f} s)")
        ax.set_xlabel("threads")
        ax.set_xticks(sorted(sub["threads"].unique()))
        ax.set_xlim(0.5, max_t + 1.5)
    axes[0].set_ylabel("speedup vs IncrSerial (median, min–max band)")
    axes[-1].legend(loc="upper left")
    fig.suptitle(f"{name}, vocab {vocab}", x=0.01, ha="left", fontsize=12)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def phases_plot(st, name, vocab, out):
    sub = st[(st["name"] == name) & (st["vocab_size"] == vocab) & (st["par_threshold_tokens"] == 4096)]
    strategies = sorted(sub["strategy"].unique(),
                        key=lambda s: (s != "serial/incr-serial", list(STRATEGY_COLORS).index(s)
                                       if s in STRATEGY_COLORS else 99))
    fig, axes = plt.subplots(1, len(strategies), figsize=(3.4 * len(strategies) + 1, 3.8),
                             sharey=True, constrained_layout=True, squeeze=False)
    for ax, s in zip(axes[0], strategies):
        d = sub[sub["strategy"] == s].sort_values("threads")
        x = np.arange(len(d))
        bottom = np.zeros(len(d))
        for j, p in enumerate(PHASES):
            v = d[f"{p}_s"].to_numpy()
            # 2px surface gap between stacked segments.
            ax.bar(x, v, bottom=bottom, width=0.7, color=SLOTS[j], edgecolor=SURFACE, lw=1,
                   label=PHASE_LABELS[p])
            bottom += v
        ax.set_xticks(x, d["threads"].astype(int))
        ax.set_xlabel("threads")
        ax.set_title(s)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("median seconds")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    axes[0][-1].legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(f"Time per phase: {name}, vocab {vocab}", x=0.01, ha="left", fontsize=12)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def amdahl_fit(threads, speedup):
    """Least-squares serial fraction f for S(p) = 1 / (f + (1 - f) / p)."""
    p, s = np.asarray(threads, float), np.asarray(speedup, float)
    # 1/S = f (1 - 1/p) + 1/p  ->  linear in f.
    a, y = 1 - 1 / p, 1 / s - 1 / p
    return float(np.clip((a @ y) / (a @ a), 0, 1)) if (a @ a) > 0 else float("nan")


def amdahl_plot(st, out):
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    i = 0
    for (name, vocab), grp in st[st["par_threshold_tokens"] == 4096].groupby(["name", "vocab_size"]):
        base = baseline(st, name, vocab)
        if base is None:
            continue
        for s, d in grp[grp["strategy"] != "serial/incr-serial"].groupby("strategy"):
            d = d.sort_values("threads")
            pc = d[d["threads"] <= P_CORES]
            if len(pc) < 3:
                continue
            sp = base["total_s"] / d["total_s"]
            f = amdahl_fit(pc["threads"], base["total_s"] / pc["total_s"])
            rows.append({"input": name, "vocab_size": vocab, "strategy": s, "serial_fraction": f,
                         "max_speedup": 1 / f if f > 0 else float("inf")})
            if i < len(SLOTS):
                color = SLOTS[i]
                ax.plot(d["threads"], sp, ls="none", marker=MARKERS[i % len(MARKERS)], ms=8,
                        color=color, mec=SURFACE, mew=2, label=f"{name} v{vocab} {s}: f={f:.3f}")
                p = np.linspace(1, d["threads"].max(), 100)
                ax.plot(p, 1 / (f + (1 - f) / p), color=color, lw=1.5)
            i += 1
    if not rows:
        plt.close(fig)
        return pd.DataFrame()
    mark_regions(ax, int(st["threads"].max()))
    ax.set_xticks(sorted(st["threads"].unique()))
    ax.set_xlabel("threads")
    ax.set_ylabel("total speedup vs IncrSerial")
    ax.set_title(f"Amdahl fit on threads 1–{P_CORES} (points: measured median; lines: fit)",
                 loc="left")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


def weak_plot(df, out):
    """Phase 1 weak scaling: efficiency t(1 thread, s bytes) / t(p threads, p*s bytes)."""
    d = df[df["pretok"] == "omp"].copy()
    d["bytes_per_thread"] = (d["input_bytes"] / d["threads"] / 1e6).round(-1)
    best = None
    for bpt, g in d.groupby("bytes_per_thread"):
        if g["threads"].nunique() >= 3 and 1 in set(g["threads"]):
            best = (bpt, g)
    if best is None:
        return False
    bpt, g = best
    med = g.groupby("threads")["pretok_s"].agg(["median", "min", "max"]).reset_index()
    t1 = med.loc[med["threads"] == 1, "median"].iloc[0]
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    mark_regions(ax, int(med["threads"].max()))
    ax.axhline(1, color=TEXT_2, lw=1)
    ax.fill_between(med["threads"], t1 / med["max"], t1 / med["min"], color=SLOTS[0], alpha=0.18,
                    lw=0)
    ax.plot(med["threads"], t1 / med["median"], color=SLOTS[0], marker="o", ms=8, mec=SURFACE,
            mew=2)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(med["threads"])
    ax.set_xlabel("threads (input grows with threads)")
    ax.set_ylabel("weak-scaling efficiency")
    ax.set_title(f"Phase 1 weak scaling, ~{bpt:.0f} MB per thread", loc="left")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def threshold_plot(st, out):
    d = st[st["merge"] == "incr-par"]
    if d["par_threshold_tokens"].nunique() < 2:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    for i, (t, g) in enumerate(d.groupby("threads")):
        g = g.sort_values("par_threshold_tokens")
        x = g["par_threshold_tokens"].clip(lower=1)
        ax.fill_between(x, g["merge_s_min"], g["merge_s_max"], color=SLOTS[i], alpha=0.18, lw=0)
        ax.plot(x, g["merge_s"], color=SLOTS[i], marker=MARKERS[i], ms=8, mec=SURFACE, mew=2,
                label=f"{t} threads")
    ax.set_xscale("log")
    ticks = sorted(d["par_threshold_tokens"].unique())
    names = {0: "0", 10**9: "∞ (all serial)"}
    ax.set_xticks([max(t, 1) for t in ticks],
                  [names.get(t, f"{t // 1024}k" if t >= 1024 else str(t)) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("par_threshold_tokens (tokens touched; below it the master merges alone)")
    ax.set_ylabel("phase 3 median seconds")
    ax.set_title("IncrPar threshold sweep", loc="left")
    ax.legend()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def md_table(df, floatfmt="{:.3f}"):
    fmt = lambda v: floatfmt.format(v) if isinstance(v, float) else str(v)  # noqa: E731
    rows = [list(df.columns), ["---"] * len(df.columns)]
    rows += [[fmt(v) for v in r] for r in df.itertuples(index=False)]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out-dir", default="results/plots")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    style()

    df, n_stalled = load(args.csv)
    keys = ["name", "vocab_size", "strategy", "pretok", "merge", "pretok_reduce",
            "par_threshold_tokens", "threads"]
    st = stats(df, keys)
    st["runs"] = df.groupby(keys).size().to_numpy()
    written = []
    for (name, vocab), _ in st.groupby(["name", "vocab_size"]):
        base = baseline(st, name, vocab)
        if base is not None and speedup_plot(st, name, vocab, base,
                                             out / f"speedup_{name}_v{vocab}.png"):
            written.append(f"speedup_{name}_v{vocab}.png")
        phases_plot(st, name, vocab, out / f"phases_{name}_v{vocab}.png")
        written.append(f"phases_{name}_v{vocab}.png")
    fits = amdahl_plot(st, out / "amdahl.png")
    if not fits.empty:
        written.append("amdahl.png")
    if weak_plot(df, out / "weak_scaling.png"):
        written.append("weak_scaling.png")
    if threshold_plot(st, out / "threshold.png"):
        written.append("threshold.png")

    power = ", ".join(sorted(df["power"].astype(str).unique()))
    lines = ["# Benchmark summary", "",
             f"Sources: {', '.join(pathlib.Path(p).name for p in args.csv)}. Power: {power}. "
             f"Warm-up runs excluded; {n_stalled} stalled run(s) dropped (wall > 2 × CPU time).", ""]
    if not fits.empty:
        lines += ["## Amdahl fit (threads 1–5)", "", md_table(fits),
                  ""]
    cols = ["name", "vocab_size", "strategy", "par_threshold_tokens", "threads", "runs"]
    for p in PHASES + ["total"]:
        st[f"{p} (med [min, max])"] = st.apply(
            lambda r: f"{r[f'{p}_s']:.3f} [{r[f'{p}_s_min']:.3f}, {r[f'{p}_s_max']:.3f}]", axis=1)
    lines += ["## All configurations (seconds)", "",
              md_table(st[cols + [c for c in st.columns if "(med" in c]]), ""]
    (out / "summary.md").write_text("\n".join(lines))
    print(f"wrote {', '.join(written + ['summary.md'])} to {out}")


if __name__ == "__main__":
    main()
