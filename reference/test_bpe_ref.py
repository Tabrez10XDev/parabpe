"""Tests for the Python reference. Run: .venv/bin/python -m unittest discover reference"""
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bpe_ref  # noqa: E402

ROOT = HERE.parent
GOLDEN = ROOT / "tests" / "golden"
A, Z = ord("a"), ord("z")

# Held-out text for the HF check: ordinary characters only, so HF's own regex engine agrees.
HELD_OUT = ("Once upon a time Lily's mom said: \"Yes, you can!\" aaaa abab 123 4.5\n\n"
            "Café naïve 日本語 Привет 👋🏽 family 👨‍👩‍👧 end.  \n  tabs\there   ")


def read_case(case_dir):
    vocab_size, specials = None, []
    for line in (case_dir / "case.txt").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(" ")
        if key == "vocab_size":
            vocab_size = int(value)
        elif key == "special":
            specials.append(value)
    return vocab_size, specials


def train_text(text, vocab_size, specials=()):
    specials = list(specials)
    return bpe_ref.train(bpe_ref.pretokenize(bpe_ref.split_docs(text, specials)), vocab_size, specials)


class MergeRules(unittest.TestCase):
    def test_aaaa_first_merge_and_rewrite(self):
        merges, vocab = train_text("aaaa", 257)
        self.assertEqual(merges, [(A, A)])
        self.assertEqual(vocab[256], b"aa")
        self.assertEqual(bpe_ref.apply_merge((A,) * 4, A, A, 256), (256, 256))

    def test_odd_run_leaves_tail(self):
        self.assertEqual(bpe_ref.apply_merge((A,) * 3, A, A, 256), (256, A))

    def test_overlapping_pairs_are_counted(self):
        self.assertEqual(bpe_ref.count_pairs({(A, A, A): 5}), Counter({(A, A): 10}))

    def test_early_stop_when_no_pairs_remain(self):
        merges, _ = train_text("ab ab", 256 + 50)
        self.assertEqual(len(merges), 2)


class TieBreak(unittest.TestCase):
    def test_tuple_order_not_joined_bytes(self):
        vocab = [bytes([i]) for i in range(256)] + [b"ab"]
        ab = 256
        pairs = Counter({(A, Z): 5, (ab, A): 5})
        # Joined bytes would pick (a,z): b"az" > b"aba". The tuple rule compares b"ab" > b"a" first.
        self.assertEqual(bpe_ref.best_pair(pairs, vocab), (ab, A))

    def test_count_beats_bytes(self):
        vocab = [bytes([i]) for i in range(256)]
        self.assertEqual(bpe_ref.best_pair(Counter({(Z, Z): 1, (A, A): 2}), vocab), (A, A))

    def test_second_element_decides_when_first_ties(self):
        vocab = [bytes([i]) for i in range(256)]
        self.assertEqual(bpe_ref.best_pair(Counter({(A, A): 3, (A, Z): 3}), vocab), (A, Z))


class SplitDocs(unittest.TestCase):
    def test_edges_and_consecutive_separators(self):
        text = "<|endoftext|>  a<|endoftext|><|endoftext|>b <|endoftext|> <|endoftext|>"
        self.assertEqual(bpe_ref.split_docs(text, ["<|endoftext|>"]), ["  a", "b ", " "])

    def test_longest_special_wins(self):
        text = "x<|endoftext|>y<|end<|endoftext|>z"
        for specials in (["<|end", "<|endoftext|>"], ["<|endoftext|>", "<|end"]):
            self.assertEqual(bpe_ref.split_docs(text, specials), ["x", "y", "z"])

    def test_no_specials_means_one_document(self):
        self.assertEqual(bpe_ref.split_docs("a<|endoftext|>b", []), ["a<|endoftext|>b"])
        self.assertEqual(bpe_ref.split_docs("", []), [])

    def test_bad_specials_rejected(self):
        with self.assertRaises(ValueError):
            bpe_ref.split_docs("a", [""])
        with self.assertRaises(ValueError):
            bpe_ref.split_docs("a", ["x", "x"])


class Formats(unittest.TestCase):
    def test_bytes_to_unicode_is_a_bijection(self):
        table = bpe_ref.bytes_to_unicode()
        self.assertEqual(sorted(table), list(range(256)))
        self.assertEqual(len(set(table.values())), 256)
        self.assertEqual(table[ord(" ")], "Ġ")
        self.assertEqual(table[ord("\n")], "Ċ")
        self.assertEqual(table[ord("a")], "a")

    def test_pattern_matches_cpp_header(self):
        header = (ROOT / "include" / "parabpe" / "types.hpp").read_text(encoding="utf-8")
        cpp = re.search(r'kGPT2Pattern =\s*R"\((.*)\)";', header).group(1)
        self.assertEqual(cpp, bpe_ref.GPT2_PAT)

    def test_empty_corpus_gives_header_only_merges(self):
        merges, vocab = train_text("", 300, ["<|endoftext|>"])
        self.assertEqual(bpe_ref.format_merges(merges, vocab), "#version: 0.2\n")

    def test_unicode_version_is_16(self):
        # SPEC.md section 2: U+10D50 is a letter from Unicode 16, U+10940 only from Unicode 17.
        self.assertTrue(bpe_ref.regex.match(r"\p{L}", "\U00010D50"))
        self.assertFalse(bpe_ref.regex.match(r"\p{L}", "\U00010940"))


class Golden(unittest.TestCase):
    def cases(self):
        dirs = sorted(p for p in GOLDEN.iterdir() if (p / "case.txt").exists())
        self.assertGreater(len(dirs), 0, "no golden cases; run reference/make_golden.py")
        return dirs

    def test_runs_case_hand_checked(self):
        lines = (GOLDEN / "runs" / "merges.txt").read_text(encoding="utf-8").splitlines()
        # 65 b's give 64 (b,b) pairs, more than any other pair; then bb+bb from the same run.
        self.assertEqual(lines[:3], ["#version: 0.2", "b b", "bb bb"])

    def test_hf_token_ids_match_reference_encode(self):
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            self.skipTest("tokenizers not installed")
        for case_dir in self.cases():
            with self.subTest(case=case_dir.name):
                vocab_size, specials = read_case(case_dir)
                text = (case_dir / "input.txt").read_bytes().decode("utf-8")
                merges, vocab = train_text(text, vocab_size, specials)
                hf = ByteLevelBPETokenizer(str(case_dir / "vocab.json"), str(case_dir / "merges.txt"),
                                           add_prefix_space=False)
                self.assertEqual(hf.encode(HELD_OUT).ids, bpe_ref.encode(HELD_OUT, merges, vocab))
                self.assertEqual(hf.decode(hf.encode(HELD_OUT).ids), HELD_OUT)


if __name__ == "__main__":
    unittest.main()
