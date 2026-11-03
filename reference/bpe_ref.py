#!/usr/bin/env python3
"""Spec-exact byte-level BPE reference (SPEC.md). Slow on purpose: easy to check by eye beats fast.

Usage:
  bpe_ref.py --input F --vocab-size N --out DIR [--special S]... [--no-special] [--words-only]

Writes words.tsv, merges.txt and vocab.json in the SPEC §6 formats. The default special token
<|endoftext|> applies only when no --special is given.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import regex

# Must equal kGPT2Pattern in include/parabpe/types.hpp (checked by test_bpe_ref.py).
GPT2_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_SPECIAL = "<|endoftext|>"
MERGES_HEADER = "#version: 0.2\n"

_PAT = regex.compile(GPT2_PAT)


def bytes_to_unicode():
    """GPT-2's byte -> printable character table (same construction as GPT-2's encoder.py)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def check_specials(specials):
    if any(not s for s in specials):
        raise ValueError("special tokens must be non-empty")
    if len(set(specials)) != len(specials):
        raise ValueError("special tokens must be unique")


def split_docs(text, specials):
    """SPEC §2: split on special tokens (leftmost match, longest first), drop them, skip empty docs."""
    check_specials(specials)
    if not specials:
        docs = [text]
    else:
        alternation = "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True))
        docs = re.split(alternation, text)
    return [d for d in docs if d]


def pretokenize(docs):
    """Phase 1: GPT-2 regex split of each doc, counted as UTF-8 bytes."""
    counts = Counter()
    for doc in docs:
        for piece in _PAT.findall(doc):
            counts[piece.encode("utf-8")] += 1
    return counts


def count_pairs(words):
    """SPEC §3: every adjacent pair, overlapping ones included, weighted by word frequency."""
    pairs = Counter()
    for word, freq in words.items():
        for pair in zip(word, word[1:]):
            pairs[pair] += freq
    return pairs


def best_pair(pairs, vocab):
    """SPEC §4: highest count; ties go to the largest (bytes(a), bytes(b)) tuple."""
    return max(pairs, key=lambda p: (pairs[p], (vocab[p[0]], vocab[p[1]])))


def apply_merge(word, a, b, new_id):
    """SPEC §3: greedy, left to right, no overlaps. (a,a,a,a) -> (z,z); (a,a,a) -> (z,a)."""
    out = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train(word_counts, vocab_size, specials):
    """Returns (merges, vocab): merges are (a, b) id pairs in order, vocab[id] is the token's bytes."""
    check_specials(specials)
    vocab = [bytes([i]) for i in range(256)] + [s.encode("utf-8") for s in specials]
    n_merges = vocab_size - len(vocab)
    if n_merges < 0:
        raise ValueError(f"vocab_size {vocab_size} < 256 + {len(specials)} special tokens")

    words = {tuple(w): f for w, f in word_counts.items()}
    seen = set(vocab[:256])
    merges = []
    for _ in range(n_merges):
        pairs = count_pairs(words)
        if not pairs:
            break  # nothing left to merge
        a, b = best_pair(pairs, vocab)
        new_id = len(vocab)
        token = vocab[a] + vocab[b]
        # Never observed (greedy merging seems to rule it out); SPEC.md would need a rule if it happens.
        assert token not in seen, f"merge {len(merges)} duplicates existing token {token!r}"
        seen.add(token)
        vocab.append(token)
        merges.append((a, b))

        merged = Counter()
        for word, freq in words.items():
            merged[apply_merge(word, a, b, new_id)] += freq
        words = merged
    return merges, vocab


def encode(text, merges, vocab):
    """Rank-based byte-level BPE encode of text with no special tokens (GPT-2 style)."""
    rank = {pair: i for i, pair in enumerate(merges)}
    first_merge_id = len(vocab) - len(merges)
    ids = []
    for piece in _PAT.findall(text):
        word = tuple(piece.encode("utf-8"))
        while len(word) > 1:
            pair = min(zip(word, word[1:]), key=lambda p: rank.get(p, len(merges)))
            if pair not in rank:
                break
            word = apply_merge(word, pair[0], pair[1], first_merge_id + rank[pair])
        ids.extend(word)
    return ids


# ---- File formats (SPEC §6) ------------------------------------------------------------

def format_words(word_counts):
    return "".join(f"{w.hex()}\t{word_counts[w]}\n" for w in sorted(word_counts))


def format_merges(merges, vocab):
    b2u = bytes_to_unicode()
    to_str = lambda i: "".join(b2u[x] for x in vocab[i])
    return MERGES_HEADER + "".join(f"{to_str(a)} {to_str(b)}\n" for a, b in merges)


def format_vocab(vocab, specials):
    """Compact JSON, ids in order, no trailing newline. The C++ writer reproduces these bytes."""
    b2u = bytes_to_unicode()
    n_special = len(specials)
    keys = [b2u[i] for i in range(256)] + list(specials)
    keys += ["".join(b2u[x] for x in tok) for tok in vocab[256 + n_special:]]
    table = {k: i for i, k in enumerate(keys)}
    assert len(table) == len(keys), "duplicate vocab.json key"
    return json.dumps(table, ensure_ascii=False, separators=(",", ":"))


def run(input_path, out_dir, vocab_size, specials, words_only=False):
    text = Path(input_path).read_bytes().decode("utf-8")  # strict: SPEC §1 promises valid UTF-8
    word_counts = pretokenize(split_docs(text, specials))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "words.tsv").write_bytes(format_words(word_counts).encode("utf-8"))
    if words_only:
        return
    merges, vocab = train(word_counts, vocab_size, specials)
    (out / "merges.txt").write_bytes(format_merges(merges, vocab).encode("utf-8"))
    (out / "vocab.json").write_bytes(format_vocab(vocab, specials).encode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int)
    ap.add_argument("--special", action="append", default=None, help="repeatable")
    ap.add_argument("--no-special", action="store_true", help="train with no special tokens")
    ap.add_argument("--words-only", action="store_true", help="stop after words.tsv")
    args = ap.parse_args()

    if args.no_special and args.special:
        ap.error("--no-special and --special are mutually exclusive")
    if args.vocab_size is None and not args.words_only:
        ap.error("--vocab-size is required unless --words-only is given")
    specials = [] if args.no_special else (args.special or [DEFAULT_SPECIAL])
    run(args.input, args.out, args.vocab_size, specials, args.words_only)


if __name__ == "__main__":
    main()
