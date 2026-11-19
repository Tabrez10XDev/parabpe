#!/usr/bin/env python3
"""Every number the report quotes, computed from results/ (S9). Writes report/numbers.tex as LaTeX macros.

    .venv/bin/python -I report/numbers.py

Rows are loaded by bench/results.py, as for every figure: warm-ups and stalls (wall > 2 x CPU + 0.05 s) dropped, and
phase 1 = chunk + pre-tokenize. Values are medians. Speedups are against serial/incr-serial at 1 thread on the same
input and vocabulary, from the same CSV. bench/plot.py instead pools that baseline over every "main" CSV, so a figure
can differ from the number quoted here by the baseline's run-to-run spread (wikitext.full v32k total: 5.53 s pooled vs
5.65 s in batch.csv, so 3.83x in the figure vs 3.91x here). Two more exceptions, each marked below: weak scaling uses
pre-tokenizer time only (as plot.py's weak_plot), and the B4 sweep numbers use bench/kill_check.py's rule (IncrSerial
with the OMP pre-tokenizer, pooled over threads).
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import results  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "report" / "numbers.tex"


def load(name):
    return results.load(RES / name)


def pick(d, **kw):
    for k, v in kw.items():
        d = d[d[k] == v]
    if d.empty:
        raise KeyError(f"no rows for {kw}")
    return d


def med(d, col, **kw):
    return float(pick(d, **kw)[col].median())


def base(d, name, vocab, col):
    return med(d, col, name=name, vocab_size=vocab, pretok="serial", merge="incr-serial", threads=1)


def speedup(d, name, vocab, col, merge, threads, pretok="omp", **kw):
    return base(d, name, vocab, col) / med(d, col, name=name, vocab_size=vocab, pretok=pretok,
                                           merge=merge, threads=threads, **kw)


N = {}


def put(key, value, fmt="{:.2f}"):
    assert re.fullmatch(r"[a-zA-Z]+", key), key
    N[key] = fmt.format(value) if not isinstance(value, str) else value


def thousands(v):
    return f"{int(round(v)):,}".replace(",", "{,}")


def main():
    b = load("batch.csv")
    for tag, name in [("wt", "wikitext.full"), ("ts", "tinystories.1G")]:
        for vtag, vocab in [("", 32000), ("C", 100000)]:
            put(f"{tag}BaseMerge{vtag}", base(b, name, vocab, "merge_s"), "{:.3f}")
            put(f"{tag}BaseMergeMs{vtag}", 1000 * base(b, name, vocab, "merge_s"), "{:.0f}")
            put(f"{tag}BaseTotal{vtag}", base(b, name, vocab, "total_s"), "{:.2f}")
            for ttag, t in [("Three", 3), ("Five", 5), ("Eleven", 11)]:
                put(f"{tag}Pone{ttag}{vtag}", speedup(b, name, vocab, "phase1_s", "incr-serial", t))
                put(f"{tag}Ser{ttag}{vtag}", speedup(b, name, vocab, "merge_s", "incr-serial", t))
                put(f"{tag}Par{ttag}{vtag}", speedup(b, name, vocab, "merge_s", "incr-par", t))
                put(f"{tag}Batch{ttag}{vtag}", speedup(b, name, vocab, "merge_s", "incr-batch", t))
                put(f"{tag}Tot{ttag}{vtag}", speedup(b, name, vocab, "total_s", "incr-batch", t))
                put(f"{tag}TotSer{ttag}{vtag}", speedup(b, name, vocab, "total_s", "incr-serial", t))
            put(f"{tag}DefaultRounds{vtag}", thousands(med(b, "rounds", name=name, vocab_size=vocab,
                                                           merge="incr-batch", threads=5)))
            put(f"{tag}ParRounds{vtag}", thousands(med(b, "parallel_rounds", name=name, vocab_size=vocab,
                                                       merge="incr-batch", threads=5)))

    # Merge time in ms for the same engine after the parallel pre-tokenizer (TinyStories parity, B8).
    for tag, name in [("wt", "wikitext.full"), ("ts", "tinystories.1G")]:
        for ttag, t in [("One", 1), ("Five", 5), ("Eleven", 11)]:
            for mtag, m in [("Ser", "incr-serial"), ("Batch", "incr-batch")]:
                put(f"{tag}Ms{mtag}{ttag}", 1000 * med(b, "merge_s", name=name, vocab_size=32000,
                                                     pretok="omp", merge=m, threads=t), "{:.1f}")

    s = load("span.csv")
    for tag, name in [("wt", "wikitext.full"), ("ts", "tinystories.1G"), ("wth", "wikitext.100M")]:
        for vtag, vocab in [("", 32000), ("C", 100000)]:
            r = pick(s, name=name, vocab_size=vocab)
            merges, rounds = float(r["merges"].median()), float(r["rounds"].median())
            put(f"{tag}Merges{vtag}", thousands(merges))
            put(f"{tag}Rounds{vtag}", thousands(rounds))
            put(f"{tag}MaxBatch{vtag}", float(r["max_batch"].max()), "{:.0f}")
            put(f"{tag}Span{vtag}", merges / rounds, "{:.1f}")

    ol = pick(load("owner_load.csv"), name="wikitext.full", vocab_size=32000)
    tl = pick(load("owner_load.csv"), name="tinystories.1G", vocab_size=32000)
    for ttag, t in [("Two", 2), ("Five", 5), ("Eleven", 11)]:
        for mtag, m in [("Cont", "contiguous"), ("Hash", "hashed")]:
            put(f"load{mtag}{ttag}", med(ol, "busiest_mean", threads=t, owner_map=m))
            put(f"tsload{mtag}{ttag}", med(tl, "busiest_mean", threads=t, owner_map=m))

    ow = load("owners.csv")
    for ttag, t in [("Five", 5), ("Eleven", 11)]:
        for mtag, m in [("Cont", "contiguous"), ("Hash", "hashed")]:
            for vtag, vocab in [("", 32000), ("C", 100000)]:
                put(f"own{mtag}{ttag}{vtag}",
                    speedup(ow, "wikitext.full", vocab, "merge_s", "incr-par", t, owner_map=m))

    rd = load("reduce.csv")
    for tag, name in [("wt", "wikitext.full"), ("ts", "tinystories.1G")]:
        for ttag, t in [("Five", 5), ("Eleven", 11)]:
            for mtag, m in [("Bucket", "bucketed"), ("Crit", "critical")]:
                put(f"{tag}Red{mtag}{ttag}", med(rd, "pretok_s", name=name, threads=t, pretok_reduce=m), "{:.2f}")

    nv = load("naive.csv")
    for tag, name in [("wt", "wikitext.10M"), ("ts", "tinystories.10M")]:
        naive1 = med(nv, "merge_s", name=name, merge="naive-serial", threads=1)
        incr1 = med(nv, "merge_s", name=name, merge="incr-serial", threads=1)
        par = pick(nv, name=name, merge="naive-par").groupby("threads")["merge_s"].median()
        put(f"{tag}NaiveSec", naive1, "{:.2f}")
        put(f"{tag}IncrSec", incr1, "{:.3f}")
        put(f"{tag}NaiveRatio", naive1 / incr1, "{:.0f}")
        put(f"{tag}NaiveParBest", naive1 / par.min(), "{:.2f}")
        put(f"{tag}NaiveParBestT", float(par.idxmin()), "{:.0f}")

    th = load("threshold.csv")
    for ttag, t in [("Five", 5), ("Eleven", 11)]:
        for ktag, k in [("Zero", 0), ("One", 1024), ("Four", 4096), ("Sixteen", 16384), ("Off", 1_000_000_000)]:
            put(f"thr{ktag}{ttag}", med(th, "merge_s", threads=t, par_threshold_tokens=k), "{:.3f}")

    wk = load("weak.csv")  # as plot.py's weak_plot: pre-tokenizer time only
    t1 = float(pick(wk, threads=1)["pretok_s"].median())
    for ttag, t in [("Five", 5), ("Eleven", 11)]:
        put(f"weak{ttag}", t1 / float(pick(wk, threads=t)["pretok_s"].median()))

    sc = load("scale.csv")
    for tag, name in [("scTs", "tinystories.full"), ("scWx", "wikitext.x6")]:
        for vtag, vocab in [("", 32000), ("C", 100000)]:
            put(f"{tag}Serial{vtag}", base(sc, name, vocab, "total_s"), "{:.1f}")
            put(f"{tag}SerialMerge{vtag}", base(sc, name, vocab, "merge_s"), "{:.3f}")
            put(f"{tag}BatchEleven{vtag}", med(sc, "total_s", name=name, vocab_size=vocab, pretok="omp",
                                               merge="incr-batch", threads=11), "{:.1f}")
            put(f"{tag}TotEleven{vtag}", speedup(sc, name, vocab, "total_s", "incr-batch", 11))
            put(f"{tag}BatchFive{vtag}", speedup(sc, name, vocab, "merge_s", "incr-batch", 5))
        put(f"{tag}GB", float(pick(sc, name=name)["input_bytes"].iloc[0]) / 1e9, "{:.1f}")

    sw = load("b4_sweep.csv")
    for vtag, vocab in [("", 32000), ("C", 100000)]:
        ser = med(sw, "merge_s", vocab_size=vocab, merge="incr-serial")  # as bench/kill_check.py
        bt = med(sw, "merge_s", vocab_size=vocab, merge="incr-batch", threads=5, owners_per_thread=1,
                 par_threshold_tokens=4096)
        put(f"killSweep{vtag}", ser / bt)

    # HF comparison (one run per row; see the profile files).
    for tag, f in [("wt", "profile_wikitext.md"), ("ts", "profile_tinystories.md")]:
        text = (RES / f).read_text(encoding="utf-8")
        hf = re.search(r"\| HF tokenizers, RAYON_NUM_THREADS=11 \| (\d+) \| ([\d.]+)", text)
        hf1 = re.search(r"\| HF tokenizers, RAYON_NUM_THREADS=1 \| (\d+) \| ([\d.]+)", text)
        ours = re.search(r"\| parabpe omp/incr-serial, 11 thread\(s\) \| (\d+) \| ([\d.]+)", text)
        put(f"{tag}HFEleven", float(hf[2]), "{:.2f}")
        put(f"{tag}HFOne", float(hf1[2]), "{:.2f}")
        put(f"{tag}OursEleven", float(ours[2]), "{:.2f}")
        put(f"{tag}HFMerges", thousands(float(hf[1])))
        put(f"{tag}OursMerges", thousands(float(ours[1])))

    # Trace (B6).
    tr = (RES / "trace_wikitext.full_v32000_t5.md").read_text(encoding="utf-8")
    busy_line = tr.split("busy time per thread (ms):")[1].splitlines()[0]
    busy = dict(re.findall(r"t(\d+) ([\d.]+)", busy_line))
    put("traceMaster", float(busy["0"]), "{:.0f}")
    put("traceWorker", sum(float(v) for k, v in busy.items() if k != "0") / (len(busy) - 1), "{:.0f}")
    sel = re.search(r"\| SELECT \| (\d+) \| ([\d.]+) \| ([\d.]+)", tr)
    put("traceSelectMean", float(sel[3]), "{:.1f}")
    oh = (RES / "trace_overhead.md").read_text(encoding="utf-8")
    m = {k: float(v) for k, v in re.findall(r"^(\w+\s+incr-batch v\d+)\s+pretok [\d.]+\s+merge ([\d.]+)", oh, re.M)}
    get = lambda pre, v: next(val for k, val in m.items() if k.split()[0] == pre and k.endswith(f"v{v}"))
    put("ohOn", 100 * (get("on", 32000) / get("new", 32000) - 1), "{:.1f}")
    put("ohOnC", 100 * (get("on", 100000) / get("new", 100000) - 1), "{:.1f}")

    # Amdahl serial fraction of the total, fitted on threads 1-5 (as plot.py's amdahl.png).
    bt = load("batch.csv")
    for vtag, vocab in [("", 32000), ("C", 100000)]:
        b0 = base(bt, "wikitext.full", vocab, "total_s")
        for mtag, merge in [("Ser", "incr-serial"), ("Batch", "incr-batch")]:
            ts = [1, 2, 3, 4, 5]
            sp = [b0 / med(bt, "total_s", name="wikitext.full", vocab_size=vocab, pretok="omp", merge=merge,
                           threads=t) for t in ts]
            put(f"wtAmdahl{mtag}{vtag}", results.amdahl_fit(ts, sp), "{:.3f}")

    # B8: merging on the performance cores (merge_threads.csv; its own IncrSerial baseline rows).
    mt = load("merge_threads.csv")
    mt["mt"] = mt["merge_threads"].astype(str).str.replace(r"\.0$", "", regex=True)
    for tag, name in [("Wt", "wikitext.full"), ("Ts", "tinystories.1G")]:
        for vtag, vocab in [("", 32000), ("C", 100000)]:
            bm, bt = base(mt, name, vocab, "merge_s"), base(mt, name, vocab, "total_s")
            for ctag, t, team in [("Five", 5, "0"), ("Eleven", 11, "0"), ("Auto", 11, "auto")]:
                kw = dict(name=name, vocab_size=vocab, pretok="omp", merge="incr-batch", threads=t, mt=team)
                put(f"mt{tag}Ms{ctag}{vtag}", 1000 * med(mt, "merge_s", **kw), "{:.1f}")
                put(f"mt{tag}Sp{ctag}{vtag}", bm / med(mt, "merge_s", **kw))
                put(f"mt{tag}Tot{ctag}{vtag}", bt / med(mt, "total_s", **kw))
                put(f"mt{tag}Pone{ctag}{vtag}", med(mt, "phase1_s", **kw), "{:.2f}")
            put(f"mt{tag}MsSer{vtag}", 1000 * bm, "{:.1f}")
            for ctag, team in [("Eleven", "0"), ("Auto", "auto")]:
                put(f"mt{tag}Par{ctag}{vtag}", bm / med(mt, "merge_s", name=name, vocab_size=vocab,
                                                       pretok="omp", merge="incr-par", threads=11, mt=team))

    # Exactness across every benchmark row (warm-ups included): one merges.txt hash per input and vocab.
    every = results.read(sorted(RES.glob("*.csv")))
    hashes = every.groupby([every["input"].map(lambda p: pathlib.Path(p).name), "vocab_size"])["merges_sha256"]
    if hashes.nunique().max() != 1:
        raise SystemExit("merges.txt differs within an input and vocabulary: the report's exactness claim fails")
    put("hashRows", thousands(len(every)))

    lines = ["% Generated by report/numbers.py from results/. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(N.items())]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(N)} numbers to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
