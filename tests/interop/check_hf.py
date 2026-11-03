#!/usr/bin/env python3
"""Interop check: our merges.txt + vocab.json must load into Hugging Face `tokenizers` and encode
held-out text to exactly the token ids that bpe_ref.encode produces from the same merges.

  .venv/bin/python tests/interop/check_hf.py --model runs/x --heldout data/wikitext.10M.txt [--max-bytes 1000000]

(A decode/encode round trip alone always passes for byte-level BPE, so token ids are compared.)
Exit code 0 = identical ids, 1 = mismatch.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reference"))
import bpe_ref  # noqa: E402


def load_model(model_dir):
    """Returns (merges as id pairs, vocab as id -> bytes) from our output files."""
    u2b = {u: b for b, u in bpe_ref.bytes_to_unicode().items()}
    table = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    lines = (model_dir / "merges.txt").read_text(encoding="utf-8").split("\n")
    if lines[0] != bpe_ref.MERGES_HEADER.rstrip("\n") or lines[-1] != "":
        raise ValueError("merges.txt: bad header or missing final newline")
    merge_lines = lines[1:-1]

    n_ids = len(table)
    first_merge = n_ids - len(merge_lines)
    vocab = [None] * n_ids
    for key, i in table.items():
        # Byte and merged tokens go through bytes_to_unicode; special tokens are literal.
        vocab[i] = key.encode("utf-8") if 256 <= i < first_merge else bytes(u2b[c] for c in key)
    if any(v is None for v in vocab):
        raise ValueError("vocab.json: ids are not 0..N-1")

    merges = []
    for i, line in enumerate(merge_lines):
        left, right = line.split(" ")  # bytes_to_unicode never produces a space
        a, b = table[left], table[right]
        if table.get(left + right) != first_merge + i:
            raise ValueError(f"merge {i} ({line!r}) does not produce id {first_merge + i}")
        merges.append((a, b))
    return merges, vocab


def first_mismatch(xs, ys):
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x != y:
            return i
    return min(len(xs), len(ys)) if len(xs) != len(ys) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="directory with merges.txt + vocab.json")
    ap.add_argument("--heldout", type=Path, required=True, help="UTF-8 text to encode")
    ap.add_argument("--max-bytes", type=int, default=1_000_000)
    args = ap.parse_args()

    from tokenizers import ByteLevelBPETokenizer

    merges, vocab = load_model(args.model)
    raw = args.heldout.read_bytes()[: args.max_bytes]
    text = raw.decode("utf-8", errors="ignore")  # drop a character cut in half at the end

    hf = ByteLevelBPETokenizer(str(args.model / "vocab.json"), str(args.model / "merges.txt"),
                               add_prefix_space=False)
    hf_ids = hf.encode(text).ids
    ref_ids = bpe_ref.encode(text, merges, vocab)

    i = first_mismatch(hf_ids, ref_ids)
    if i is not None:
        print(f"MISMATCH at token {i}: hf={hf_ids[i:i + 8]} ref={ref_ids[i:i + 8]}")
        print("  hf  :", [hf.id_to_token(t) for t in hf_ids[max(0, i - 3):i + 5]])
        return 1
    if hf.decode(hf_ids) != text:
        print("MISMATCH: HF decode does not reproduce the text")
        return 1
    used = sum(1 for t in ref_ids if t >= len(vocab) - len(merges))
    print(f"ok: {len(ref_ids):,} tokens identical ({used:,} merged tokens) on {len(raw):,} bytes; "
          f"{len(merges)} merges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
