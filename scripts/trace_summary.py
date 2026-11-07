#!/usr/bin/env python3
"""Check and summarise a `parabpe train --trace` file (Chrome Trace Event JSON, B6).

    python3 -I scripts/trace_summary.py runs/x/trace.json [--markdown out.md]

Checks: the file parses, every complete event ("ph": "X") has a known name, a thread inside the
run's thread count and a duration >= 0, nothing was dropped, and on each thread the events nest
or are disjoint (a span never half-overlaps another). Prints, per event kind, the count, the
total time and the share of that time sampled on E-cores. The core is sampled once at each
event's start and macOS has no thread pinning, so the E-core share is an estimate.

Exit codes: 0 valid, 1 invalid.
"""

import argparse
import collections
import json
import sys

KINDS = ["CHUNK", "PRETOK_UNIT", "SHARD_REDUCE", "SELECT", "SERIAL_APPLY", "OWNER_MERGE",
         "ZERO_SWEEP", "ORDER_UPDATE"]
PHASE3 = ("OWNER_MERGE", "SERIAL_APPLY", "SELECT", "ORDER_UPDATE", "ZERO_SWEEP")
EPS_US = 0.01  # timestamps are printed to 1 ns; allow rounding when checking nesting


def check(trace):
    errors = []
    other = trace.get("otherData", {})
    threads = other.get("threads")
    if not isinstance(threads, int) or threads < 1:
        errors.append("otherData.threads missing")
        threads = 1 << 30
    if other.get("dropped", 0):
        errors.append(f"{other['dropped']} events dropped")
    spans = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]
    if other.get("events") is not None and other["events"] != len(spans):
        errors.append(f"otherData.events {other['events']} != {len(spans)} complete events")
    by_tid = collections.defaultdict(list)
    for e in spans:
        if e.get("name") not in KINDS:
            errors.append(f"unknown event name {e.get('name')!r}")
        if not 0 <= e.get("tid", -1) < threads:
            errors.append(f"tid {e.get('tid')} outside 0..{threads - 1}")
        if e.get("dur", -1) < 0:
            errors.append(f"negative duration in {e}")
        by_tid[e.get("tid")].append(e)
    for tid, events in by_tid.items():
        # Longest first at equal starts, so a parent precedes the spans it contains.
        events.sort(key=lambda e: (e["ts"], -e["dur"]))
        stack = []
        for e in events:
            while stack and e["ts"] >= stack[-1] - EPS_US:
                stack.pop()
            end = e["ts"] + e["dur"]
            if stack and end > stack[-1] + EPS_US:
                errors.append(f"tid {tid}: {e['name']} at {e['ts']} overlaps its enclosing span")
                break
            stack.append(end)
    return spans, errors


def summarise(trace, spans):
    other = trace.get("otherData", {})
    rows = collections.OrderedDict((k, [0, 0.0, 0.0]) for k in KINDS)
    for e in spans:
        r = rows[e["name"]]
        r[0] += 1
        r[1] += e["dur"]
        if e.get("args", {}).get("core") == "E":
            r[2] += e["dur"]
    labelled = "e_cores" in other
    lines = [f"Trace: {other.get('threads')} threads, {len(spans)} events. "
             + (f"Cores: {other['core_rule']}." if labelled else "No core labels on this OS."),
             "",
             "| event | count | total ms | mean µs | E-core share |",
             "|---|---:|---:|---:|---:|"]
    for name, (n, us, e_us) in rows.items():
        if not n:
            continue
        share = f"{100 * e_us / us:.0f}%" if labelled and us > 0 else "n/a"
        lines.append(f"| {name} | {n} | {us / 1000:.1f} | {us / n:.1f} | {share} |")
    busy = collections.Counter()
    for e in spans:
        if e["name"] in PHASE3:
            busy[e["tid"]] += e["dur"]
    if busy:
        lines += ["", "Phase 3 busy time per thread (ms): "
                  + ", ".join(f"t{t} {busy[t] / 1000:.1f}" for t in sorted(busy))]
        if labelled:
            e_busy = collections.Counter()
            for e in spans:
                if e["name"] in PHASE3 and e.get("args", {}).get("core") == "E":
                    e_busy[e["tid"]] += e["dur"]
            lines += ["", "Phase 3 E-core share per thread: "
                      + ", ".join(f"t{t} {100 * e_busy[t] / busy[t]:.0f}%" for t in sorted(busy))]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("trace")
    p.add_argument("--markdown", help="also write the summary to this file")
    args = p.parse_args()
    with open(args.trace, encoding="utf-8") as f:
        trace = json.load(f)
    spans, errors = check(trace)
    for err in errors[:20]:
        print(f"INVALID: {err}")
    if errors:
        return 1
    text = summarise(trace, spans)
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
