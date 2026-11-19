#!/usr/bin/env python3
"""bench/results.py: the row rules every figure and report number relies on.

    .venv/bin/python -I tests/bench/test_results.py
"""

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
try:
    import results  # noqa: E402  (needs pandas: the repo's .venv)
except ImportError as e:
    print(f"skip: {e} (create .venv from requirements.txt)")
    sys.exit(77)

HEADER = "input,warmup,threads,chunk_s,pretok_s,total_s,cpu_s,power\n"
ROWS = [
    "data/wikitext.full.txt,1,1,0.1,4.0,5.0,5.0,AC Power\n",   # warm-up
    "data/wikitext.full.txt,0,1,0.1,4.0,5.0,5.0,AC Power\n",   # kept
    "data/wikitext.full.txt,0,4,0.1,1.0,1.5,6.0,n/a\n",        # kept: n/a stays a string
    "data/wikitext.full.txt,0,4,0.1,1.0,9.0,1.0,AC Power\n",   # stalled: 9.0 > 2 x 1.0 + 0.05
    "data/tinystories.1G.txt,0,2,0.2,2.0,2.1,1.0,AC Power\n",  # stalled: 2.1 > 2 x 1.0 + 0.05
]


class Rules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv = pathlib.Path(self.tmp.name) / "x.csv"
        self.csv.write_text(HEADER + "".join(ROWS))

    def tearDown(self):
        self.tmp.cleanup()

    def test_drops_warmups_and_stalls(self):
        df, n_stalled = results.clean(results.read([self.csv]))
        self.assertEqual(n_stalled, 2)  # the 9.0 s row and the 2.1 s row (limit 2.05 s)
        self.assertEqual(len(df), 2)
        self.assertEqual(sorted(df["threads"]), [1, 4])

    def test_names_phase1_and_na(self):
        df = results.load(self.csv)
        self.assertEqual(set(df["name"]), {"wikitext.full"})
        self.assertAlmostEqual(float(df[df["threads"] == 1]["phase1_s"].iloc[0]), 4.1)
        self.assertIn("n/a", set(df["power"]))  # not NaN: kill_check accepts it as off-macOS

    def test_stall_limit(self):
        df = results.read([self.csv])
        self.assertEqual(list(results.stalled(df)), [False, False, False, True, True])


if __name__ == "__main__":
    unittest.main()
