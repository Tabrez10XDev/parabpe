#!/usr/bin/env python3
"""Regenerates tests/golden/ from the inline cases below, using bpe_ref.py as the oracle.

  make_golden.py            write tests/golden/<case>/{input.txt,case.txt,words.tsv,merges.txt,vocab.json}
                            and tests/golden/unicode_classes.txt
  make_golden.py --check    regenerate into a temp dir and compare sha256 with what is on disk (exit 1 on drift)

case.txt is line-based so the C++ tests can read it without a JSON library:
  vocab_size <N>
  special <token>        (one line per special token, in config order; none = no special tokens)
"""
import argparse
import hashlib
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import regex

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpe_ref  # noqa: E402

GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden"
EOT = "<|endoftext|>"

ENGLISH = (
    "Once upon a time, there was a little girl named Lily. She loved to play outside in the sun.\n"
    "One day, she saw a big, red ball. \"Can I play with it?\" she asked her mom. "
    "Her mom said, \"Yes, you can!\" Lily's friend Tom didn't want to share, but they'll learn.\n\n"
    "The end. It's 2024 and we've got 12 apples, 3.5 pears and 1,000 grapes...\n"
)


def case(text, vocab_extra, specials=(EOT,)):
    return {"text": text, "vocab_size": 256 + len(specials) + vocab_extra, "specials": list(specials)}


CASES = {
    "unicode": case(
        EOT.join([
            "Café déjà vu, naïve façade, Ångström, Straße, ÆØÅ æøå. " * 3,
            "日本語のテキストです。東京は大きい都市です。日本語 日本語 日本語。" * 3,
            "Привет, как дела? Всё хорошо, спасибо! Привет привет. " * 3,
            "مرحبا بالعالم، كيف حالك؟ مرحبا مرحبا. " * 2,
            "नमस्ते दुनिया, आप कैसे हैं? नमस्ते नमस्ते। " * 2,
            "Ελληνικά: γειά σου κόσμε. Ελληνικά ελληνικά. " * 2,
            "한국어 텍스트입니다. 한국어 한국어. " * 2,
        ]),
        80,
    ),
    "emoji": case(
        EOT.join([
            "hi 👋 hi 👋🏽 hi 👋🏿 wave 👋👋👋 " * 3,
            "family 👨\u200d👩\u200d👧\u200d👦 and 👩\u200d❤\ufe0f\u200d👨 and 🏳\ufe0f\u200d🌈 flags 🇮🇳🇯🇵🇺🇸 " * 3,
            "mixed😀text😀with😀emoji 🎉🎉 party 🎉 " * 3,
        ]),
        60,
    ),
    "runs": case(
        EOT.join([
            "aaaa",
            "aaaaaaa aaa aaaaa a aa",
            "abababab ababab abab",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            " aaaa\n\n\n\n\n\n        end      ",
            "!!!!!!!!!!!!!!!!!!!!!!!! ???? ....... ------------",
            "1111111111 1212121212 0000",
        ]),
        50,
    ),
    "ties": case(
        # Every two-letter combination once: most pair counts tie, so the tie-break decides.
        " ".join(a + b for a in "abcdefgh" for b in "abcdefgh")
        + EOT + " ".join(a + b + c for a in "xyz" for b in "xyz" for c in "xyz")
        + EOT + "ab ba ab ba abc bca cab",
        60,
    ),
    "multi_occurrence": case(
        EOT.join([
            "abcabcabcabc abcabc abc " * 3,
            "banana bananas bandana " * 4,
            "xyxyxyxyxy yxyxyx " * 3,
            "the theme thesis there then " * 3,
            "aaaaab baaaaa abaaba " * 2,
        ]),
        60,
    ),
    "doc_edges": case(
        # Whitespace at doc start/end, empty docs (consecutive separators), whitespace-only docs.
        EOT + "  leading spaces here" + EOT + EOT + "trailing spaces   " + EOT
        + "\n\nnewlines first" + EOT + "   " + EOT + "\n" + EOT + "tab\tsep\t" + EOT
        + " " + EOT + "x" + EOT + "end of corpus  \n" + EOT,
        40,
    ),
    "no_separator": case(ENGLISH * 3, 100),
    "multi_special": case(
        # "<|end" is a prefix of "<|endoftext|>": leftmost match, longest first, must win.
        "first doc<|endoftext|>second<|end>third<|end<|endoftext|>fourth "
        "<|endoftext|><|end|>fifth <|endo|> not special<|end",
        40,
        specials=(EOT, "<|end"),
    ),
    "early_stop": case("ab ab", 50),  # only 2 merges are possible: (a,b) then (Ġ,ab)
    "whitespace_zoo": case(
        EOT.join([
            "line one\r\nline two\r\n\r\nline three\r\n",
            "tab\tsep\t\tdouble\vvertical\fformfeed",
            "info\x1csep\x1dgroup\x1erecord\x1funit\x85nextline",
            "nb\u00a0space line\u2028sep para\u2029sep ideo\u3000space zero\u200bwidth",
            "mongolian \u180e vowel\u180esep \u180e\u180e x",  # raw PCRE2 \s matches U+180E; Unicode 16 does not
            "\ufeffbom at start",
            "it's IT'S It'S they’re we’ve 's 'S",
            "cafe\u0301 vs café n\u0303",
            "½ ¾ ² ³ ¹ Ⅻ ①②",
            "fullwidth １２３ ａｂｃ arabic ٣٤٥",
            "trailing runs   \n\n   ",
            "  \t \n ",
        ]),
        60,
    ),
    "unicode16": case(
        # U+10D50.. (Garay letters) and U+1E5D0.. (Ol Onal) are letters only from Unicode 16;
        # U+10D40.. are Garay digits; U+10940 (Sidetic) is a letter only from Unicode 17.
        EOT.join([
            "garay \U00010D50\U00010D51\U00010D52 \U00010D50\U00010D51 x\U00010D53y",
            "olonal \U0001E5D0\U0001E5D1\U0001E5D2 \U0001E5D0\U0001E5D1",
            "digits \U00010D40\U00010D41\U00010D42 7\U00010D43",
            "sidetic \U00010940\U00010941 a\U00010940b",
        ]),
        30,
    ),
}


def write_case(out_dir, spec):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "input.txt").write_bytes(spec["text"].encode("utf-8"))
    lines = [f"vocab_size {spec['vocab_size']}"] + [f"special {s}" for s in spec["specials"]]
    (out_dir / "case.txt").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    bpe_ref.run(out_dir / "input.txt", out_dir, spec["vocab_size"], spec["specials"])


def write_unicode_classes(path):
    """Code-point ranges of the three classes the GPT-2 pattern uses, for the C++ parity test."""
    classes = {"s": regex.compile(r"\s"), "L": regex.compile(r"\p{L}"), "N": regex.compile(r"\p{N}")}
    lines = [f"# Python regex=={version('regex')} (SPEC.md section 2: Unicode 16.0). "
             "Format: <class> <first hex> <last hex>, inclusive; surrogates skipped."]
    for name, pat in classes.items():
        start = prev = None
        for cp in range(0x110000):
            if 0xD800 <= cp <= 0xDFFF or not pat.match(chr(cp)):
                continue
            if start is not None and cp == prev + 1:
                prev = cp
                continue
            if start is not None:
                lines.append(f"{name} {start:04X} {prev:04X}")
            start = prev = cp
        lines.append(f"{name} {start:04X} {prev:04X}")
    path.write_bytes(("\n".join(lines) + "\n").encode("ascii"))


def generate(root):
    for name, spec in CASES.items():
        write_case(root / name, spec)
    write_unicode_classes(root / "unicode_classes.txt")


def digests(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != ".gitkeep"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare a fresh build with tests/golden/")
    args = ap.parse_args()

    if not args.check:
        generate(GOLDEN)
        print(f"wrote {len(CASES)} cases + unicode_classes.txt to {GOLDEN}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        generate(Path(tmp))
        fresh, on_disk = digests(Path(tmp)), digests(GOLDEN)
    drift = sorted(k for k in fresh.keys() | on_disk.keys() if fresh.get(k) != on_disk.get(k))
    for k in drift:
        print(f"DRIFT {k}")
    print("golden files up to date" if not drift else f"{len(drift)} file(s) differ")
    sys.exit(1 if drift else 0)


if __name__ == "__main__":
    main()
