#!/usr/bin/env python3
"""ParaBPE Studio server (B7): standard library only, bound to 127.0.0.1.

    python3 -I studio/serve.py [--port 8765]     # then open http://127.0.0.1:8765/

It serves the viewer (studio/index.html, studio.js, studio.css), lists the corpora directly under
data/, runs `parabpe train --trace` on one of them, and serves that run's trace and merges.txt
together with the sha256 of an IncrSerial reference run (cached per input and vocab size). It
lists finished runs for the Race (B9), serves results/*.csv for the benchmark charts, and runs
tests/interop/check_hf.py to encode a sentence with Hugging Face `tokenizers`.

Safety: inputs must be *.txt files directly under data/ (resolved paths, no traversal), every
command is an argument list (no shell), nothing from data/ is executed, POSTs must be JSON, and
the Host header must name this server (a guard against DNS rebinding).
"""

import argparse
import hashlib
import http.server
import json
import pathlib
import re
import secrets
import socketserver
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUDIO = ROOT / "studio"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RUNS = ROOT / "runs" / "studio"
VENV_PY = ROOT / ".venv" / "bin" / "python"
STATIC = {"/": "index.html", "/index.html": "index.html", "/studio.js": "studio.js",
          "/studio.css": "studio.css"}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8"}
MERGES = ["incr-batch", "incr-par", "incr-serial"]
RUN_ID = re.compile(r"[0-9a-f]{12}")
CSV_NAME = re.compile(r"[a-z0-9_]+\.csv")
TIMEOUT_S = 900
MAX_TEXT = 1000

run_lock = threading.Lock()  # one training run at a time: runs are timed


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def list_inputs():
    if not DATA.is_dir():
        return []
    return [{"name": p.name, "bytes": p.stat().st_size}
            for p in sorted(DATA.glob("*.txt")) if p.is_file()]


def resolve_input(name):
    """data/<name>, only for a *.txt file directly under data/ (raises ValueError otherwise)."""
    if not isinstance(name, str) or "/" in name or "\\" in name or not name.endswith(".txt"):
        raise ValueError("input must be the name of a .txt file in data/")
    path = (DATA / name).resolve()
    if path.parent != DATA.resolve() or not path.is_file():
        raise ValueError(f"no such input in data/: {name}")
    return path


def int_field(body, key, lo, hi):
    value = body.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi:
        raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
    return value


def merge_threads_args(body, merge):
    """["--merge-threads", "auto" | "N"] for a parallel engine (B9), [] when absent, 0 or incr-serial."""
    value = body.get("merge_threads", 0)
    if value is None or value == 0 or merge == "incr-serial":
        return []
    if value == "auto":
        return ["--merge-threads", "auto"]
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 64:
        return ["--merge-threads", str(value)]
    raise ValueError('merge_threads must be "auto" or an integer in [1, 64]')


def train(binary, path, vocab, threads, merge, out, extra=()):
    cmd = [str(binary), "train", "--input", str(path), "--vocab-size", str(vocab),
           "--threads", str(threads), "--pretok", "omp" if threads > 1 else "serial",
           "--merge", merge, "--out", str(out), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    if proc.returncode != 0:
        raise RuntimeError(f"parabpe failed: {proc.stderr.strip()[-500:]}")
    return json.loads((out / "timings.json").read_text(encoding="utf-8"))


def reference(binary, path, vocab):
    """sha256 of IncrSerial's merges.txt (serial pre-tokenizer, 1 thread), cached by input."""
    stat = path.stat()
    key = f"{path.name}_{vocab}"
    out = RUNS / "ref" / key
    meta = out / "ref.json"
    if meta.is_file():
        cached = json.loads(meta.read_text(encoding="utf-8"))
        if cached.get("size") == stat.st_size and cached.get("mtime") == stat.st_mtime:
            return cached["sha256"], True
    out.mkdir(parents=True, exist_ok=True)
    train(binary, path, vocab, 1, "incr-serial", out)
    digest = sha256(out / "merges.txt")
    meta.write_text(json.dumps({"sha256": digest, "size": stat.st_size, "mtime": stat.st_mtime}),
                    encoding="utf-8")
    return digest, False


def do_run(binary, body):
    path = resolve_input(body.get("input"))
    vocab = int_field(body, "vocab_size", 257, 300_000)
    threads = int_field(body, "threads", 1, 64)
    merge = body.get("merge")
    if merge not in MERGES:
        raise ValueError(f"merge must be one of {MERGES}")
    extra = ["--batch-full-rounds"] if merge == "incr-batch" and body.get("full_rounds") else []
    merge_threads = merge_threads_args(body, merge)
    extra += merge_threads
    run_id = secrets.token_hex(6)
    out = RUNS / run_id
    out.mkdir(parents=True)
    with run_lock:
        timings = train(binary, path, vocab, threads, merge, out,
                        [*extra, "--trace", str(out / "trace.json")])
        ref_sha, cached = reference(binary, path, vocab)
    digest = sha256(out / "merges.txt")
    summary = {"id": run_id, "input": path.name, "vocab_size": vocab, "threads": threads,
               "merge": merge, "full_rounds": "--batch-full-rounds" in extra,
               "merge_threads": merge_threads[1] if merge_threads else 0,
               "merge_threads_used": timings.get("config", {}).get("merge_threads", threads),
               "timings": timings,
               "sha256": digest, "reference_sha256": ref_sha, "reference_cached": cached,
               "identical": digest == ref_sha, "finished": time.time()}
    (out / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def list_runs(limit=50):
    """Finished runs (their run.json summaries), newest first. Unreadable files are skipped."""
    runs = []
    for p in RUNS.glob("*/run.json"):
        if not RUN_ID.fullmatch(p.parent.name):
            continue
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    runs.sort(key=lambda r: r.get("finished", 0), reverse=True)
    return runs[:limit]


def latest_run():
    runs = list_runs(1)
    return runs[0] if runs else None


def encode(body):
    run_id, text = body.get("id"), body.get("text")
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id) or not (RUNS / run_id).is_dir():
        raise ValueError("unknown run id")
    if not isinstance(text, str) or not 0 < len(text) <= MAX_TEXT:
        raise ValueError(f"text must be 1-{MAX_TEXT} characters")
    if not VENV_PY.is_file():
        raise RuntimeError("needs .venv with `tokenizers` (see requirements.txt)")
    proc = subprocess.run([str(VENV_PY), "-I", str(ROOT / "tests" / "interop" / "check_hf.py"),
                           "--model", str(RUNS / run_id), "--text", text],
                          capture_output=True, text=True, timeout=120)
    lines = proc.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError(f"check_hf.py failed: {proc.stderr.strip()[-500:]}")
    return json.loads(lines[-1])


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "ParaBPEStudio/1"
    binary = ROOT / "build" / "parabpe"
    port = 8765

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def host_ok(self):
        host = self.headers.get("Host", "")
        return host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def send(self, status, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path, ctype):
        self.send(200, path.read_bytes(), ctype)

    def do_GET(self):
        if not self.host_ok():
            return self.send(403, {"error": "bad Host header"})
        route = self.path.split("?", 1)[0]
        if route in STATIC:
            path = STUDIO / STATIC[route]
            return self.send_file(path, TYPES[path.suffix])
        if route == "/api/inputs":
            return self.send(200, {"inputs": list_inputs(), "binary": self.binary.is_file(),
                                   "hf": VENV_PY.is_file()})
        if route == "/api/results":
            files = sorted(p.name for p in RESULTS.glob("*.csv")) if RESULTS.is_dir() else []
            return self.send(200, {"files": files})
        if route.startswith("/api/results/"):
            name = route[len("/api/results/"):]
            if CSV_NAME.fullmatch(name) and (RESULTS / name).is_file():
                return self.send_file(RESULTS / name, "text/csv; charset=utf-8")
            return self.send(404, {"error": "no such results file"})
        if route == "/api/runs":
            return self.send(200, {"runs": list_runs()})
        if route == "/api/runs/latest":
            return self.send(200, {"run": latest_run()})
        m = re.fullmatch(r"/api/runs/([0-9a-f]{12})/(trace\.json|merges\.txt|run\.json)", route)
        if m and (RUNS / m[1] / m[2]).is_file():
            ctype = ("text/plain; charset=utf-8" if m[2].endswith(".txt")
                     else "application/json; charset=utf-8")
            return self.send_file(RUNS / m[1] / m[2], ctype)
        return self.send(404, {"error": "not found"})

    def do_POST(self):
        if not self.host_ok():
            return self.send(403, {"error": "bad Host header"})
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self.send(415, {"error": "POST a JSON body"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1 << 16:
                raise ValueError("bad body length")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            if self.path == "/api/run":
                return self.send(200, do_run(self.binary, body))
            if self.path == "/api/encode":
                return self.send(200, encode(body))
            return self.send(404, {"error": "not found"})
        except (ValueError, json.JSONDecodeError) as e:
            return self.send(400, {"error": str(e)})
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            return self.send(500, {"error": str(e)})


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        # HTTPServer.server_bind calls socket.getfqdn(), a reverse DNS lookup that can stall for
        # half a minute on some networks; the name is never used here.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--binary", type=pathlib.Path, default=Handler.binary)
    args = p.parse_args()
    Handler.binary = args.binary.resolve()
    Handler.port = args.port
    RUNS.mkdir(parents=True, exist_ok=True)
    server = Server(("127.0.0.1", args.port), Handler)
    print(f"ParaBPE Studio on http://127.0.0.1:{args.port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
