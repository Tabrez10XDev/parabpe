#!/usr/bin/env python3
"""Studio server checks (B7): input confinement to data/, request validation, the Host guard.
No training runs: the handler is exercised against a temporary data/ and results/ directory.

    python3 -I tests/studio/test_serve.py
"""

import http.client
import importlib.util
import json
import pathlib
import tempfile
import threading
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("serve", ROOT / "studio" / "serve.py")
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


class Confinement(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self.tmp.name)
        self.old = serve.DATA, serve.RESULTS, serve.RUNS
        serve.DATA, serve.RESULTS, serve.RUNS = base / "data", base / "results", base / "runs"
        for d in (serve.DATA, serve.RESULTS, serve.RUNS, serve.DATA / "raw"):
            d.mkdir(parents=True)
        (serve.DATA / "ok.txt").write_text("hello")
        (serve.DATA / "raw" / "nested.txt").write_text("x")
        (base / "outside.txt").write_text("secret")
        (serve.DATA / "link.txt").symlink_to(base / "outside.txt")
        (serve.RESULTS / "batch.csv").write_text("a,b\n1,2\n")

    def tearDown(self):
        serve.DATA, serve.RESULTS, serve.RUNS = self.old
        self.tmp.cleanup()

    def test_accepts_a_txt_file_directly_under_data(self):
        self.assertEqual(serve.resolve_input("ok.txt"), (serve.DATA / "ok.txt").resolve())

    def test_rejects_everything_else(self):
        for name in ["../outside.txt", "raw/nested.txt", "/etc/passwd", "link.txt", "missing.txt",
                     "ok", "", None, 3, "..\\outside.txt", "raw"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                serve.resolve_input(name)

    def test_lists_only_top_level_txt(self):
        names = [i["name"] for i in serve.list_inputs()]
        self.assertIn("ok.txt", names)
        self.assertNotIn("nested.txt", names)

    def test_int_fields(self):
        self.assertEqual(serve.int_field({"t": 4}, "t", 1, 64), 4)
        for bad in [0, 65, "4", 4.0, True, None]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                serve.int_field({"t": bad}, "t", 1, 64)

    def test_run_rejects_bad_requests_before_running(self):
        good = {"input": "ok.txt", "vocab_size": 1000, "threads": 2, "merge": "incr-batch"}
        for key, bad in [("input", "../outside.txt"), ("vocab_size", 10), ("threads", 0),
                         ("merge", "naive-serial"), ("merge", "incr-batch; rm -rf /")]:
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                serve.do_run(pathlib.Path("/nonexistent/parabpe"), {**good, key: bad})
        self.assertEqual(list(serve.RUNS.iterdir()), [])  # nothing was started

    def test_merge_threads_argument(self):
        self.assertEqual(serve.merge_threads_args({}, "incr-batch"), [])
        self.assertEqual(serve.merge_threads_args({"merge_threads": 0}, "incr-batch"), [])
        self.assertEqual(serve.merge_threads_args({"merge_threads": "auto"}, "incr-batch"),
                         ["--merge-threads", "auto"])
        self.assertEqual(serve.merge_threads_args({"merge_threads": 7}, "incr-par"),
                         ["--merge-threads", "7"])
        self.assertEqual(serve.merge_threads_args({"merge_threads": "auto"}, "incr-serial"), [])
        for bad in ["rm -rf /", "5", 65, -1, 2.0, True, [5], "AUTO"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                serve.merge_threads_args({"merge_threads": bad}, "incr-batch")

    def test_run_rejects_bad_merge_threads_before_running(self):
        good = {"input": "ok.txt", "vocab_size": 1000, "threads": 2, "merge": "incr-batch"}
        with self.assertRaises(ValueError):
            serve.do_run(pathlib.Path("/nonexistent/parabpe"), {**good, "merge_threads": "auto; rm -rf /"})
        self.assertEqual(list(serve.RUNS.iterdir()), [])

    def test_lists_runs_newest_first_and_skips_junk(self):
        for rid, finished in [("0123456789ab", 10), ("0123456789ac", 30), ("0123456789ad", 20)]:
            (serve.RUNS / rid).mkdir()
            (serve.RUNS / rid / "run.json").write_text(json.dumps({"id": rid, "finished": finished}))
        (serve.RUNS / "not-a-run-id").mkdir()
        (serve.RUNS / "not-a-run-id" / "run.json").write_text("{}")
        (serve.RUNS / "0123456789ae").mkdir()
        (serve.RUNS / "0123456789ae" / "run.json").write_text("{ broken")
        self.assertEqual([r["id"] for r in serve.list_runs()],
                         ["0123456789ac", "0123456789ad", "0123456789ab"])
        self.assertEqual([r["id"] for r in serve.list_runs(2)], ["0123456789ac", "0123456789ad"])
        self.assertEqual(serve.latest_run()["id"], "0123456789ac")

    def test_encode_rejects_unknown_runs_and_long_text(self):
        for body in [{"id": "../../etc", "text": "x"}, {"id": "0123456789ab", "text": "x"}]:
            with self.subTest(body=body), self.assertRaises(ValueError):
                serve.encode(body)
        (serve.RUNS / "0123456789ab").mkdir()
        with self.assertRaises(ValueError):
            serve.encode({"id": "0123456789ab", "text": "x" * (serve.MAX_TEXT + 1)})


class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serve.Server(("127.0.0.1", 0), serve.Handler)
        serve.Handler.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", serve.Handler.port, timeout=10)
        conn.request(method, path, body=body, headers=headers or {})
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, data

    def test_serves_the_viewer_only(self):
        self.assertEqual(self.request("GET", "/")[0], 200)
        self.assertEqual(self.request("GET", "/studio.js")[0], 200)
        for path in ["/serve.py", "/../SPEC.md", "/api/results/..%2FSPEC.md", "/api/runs/x/trace.json"]:
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)
        status, data = self.request("GET", "/api/runs")
        self.assertEqual(status, 200)
        self.assertIn("runs", json.loads(data))

    def test_host_guard(self):
        status, _ = self.request("GET", "/api/inputs", headers={"Host": "evil.example"})
        self.assertEqual(status, 403)

    def test_post_must_be_json(self):
        status, _ = self.request("POST", "/api/run", body="input=x",
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(status, 415)
        status, data = self.request("POST", "/api/run", body=json.dumps({"input": "../x.txt"}),
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400, data)


if __name__ == "__main__":
    unittest.main()
