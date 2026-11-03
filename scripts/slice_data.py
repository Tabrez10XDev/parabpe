#!/usr/bin/env python3
"""Normalizes a raw corpus (SPEC.md §1) and writes size slices made of whole documents.

  .venv/bin/python -I scripts/slice_data.py --source tinystories --sizes 10M,100M,1G --no-full

Reads data/raw/<source>/, writes data/<source>.<size>.txt (and data/<source>.full.txt unless --no-full),
then updates data/CHECKSUMS.sha256 (`shasum -a 256 -c` format, paths relative to data/).

- Invalid UTF-8 becomes U+FFFD, so every output is valid UTF-8.
- Documents are joined with <|endoftext|> (no leading or trailing separator); empty documents are dropped.
  TinyStories already has separators. WikiText (HF parquet rows are lines, "" is a blank line) starts an
  article at a line ` = Title = `. enwik9 starts a document at each `<page>`; the `<mediawiki>` header
  before the first page is document 0.
- A slice is the longest prefix of whole documents whose size is at or below the limit (K/M/G = 10^3/10^6/10^9 bytes).
Downloaded data is untrusted: it is only read and decoded, never executed. Run with `python -I`.
"""
import argparse
import codecs
import hashlib
import os
import re
import sys
from pathlib import Path

EOT = "<|endoftext|>"
EOT_BYTES = len(EOT.encode("utf-8"))
CHUNK = 1 << 22
UNITS = {"K": 10**3, "M": 10**6, "G": 10**9}
WIKI_TITLE = re.compile(r"^ = [^=].* = $")


def parse_size(s):
    m = re.fullmatch(r"(\d+)([KMG])", s)
    if not m:
        raise argparse.ArgumentTypeError(f"bad size {s!r} (use e.g. 10M, 1G)")
    return s, int(m.group(1)) * UNITS[m.group(2)]


def decoded_chunks(path):
    """Streams a file as text; invalid UTF-8 becomes U+FFFD, even across chunk boundaries."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            yield decoder.decode(block)
    yield decoder.decode(b"", final=True)


def split_on(chunks, separator):
    """Documents between separators; the separators are dropped."""
    buf = ""
    for chunk in chunks:
        buf += chunk
        parts = buf.split(separator)
        buf = parts.pop()
        yield from parts
    yield buf


def split_before(chunks, marker):
    """Documents that each start at `marker` (text before the first marker is its own document)."""
    buf = ""
    for chunk in chunks:
        buf += chunk
        start = 0  # start of the current (unfinished) document in buf
        while (i := buf.find(marker, start + 1)) > 0:
            yield buf[start:i]
            start = i
        buf = buf[start:]  # the tail may hold a marker that straddles the next chunk
    yield buf


def tinystories_docs(raw):
    return split_on(decoded_chunks(raw / "TinyStoriesV2-GPT4-train.txt"), EOT)


def enwik9_docs(raw):
    return split_before(decoded_chunks(raw / "enwik9"), "<page>")


def wikitext_docs(raw):
    import pyarrow.parquet as pq

    doc = []
    for shard in ("train-00000-of-00002.parquet", "train-00001-of-00002.parquet"):
        for batch in pq.ParquetFile(raw / shard).iter_batches(columns=["text"]):
            for line in batch.column(0).to_pylist():
                # Blank lines arrive as "" rows; any blank rows before the first title stay with that article.
                if WIKI_TITLE.match(line.rstrip("\n")) and any(l != "\n" for l in doc):
                    yield "".join(doc)
                    doc = []
                doc.append(line or "\n")
    yield "".join(doc)


SOURCES = {"tinystories": tinystories_docs, "wikitext": wikitext_docs, "enwik9": enwik9_docs}


class Output:
    """One output file. Writes to <name>.part and renames on close; stops at the first doc that does not fit."""

    def __init__(self, path, limit):
        self.path, self.limit, self.size, self.docs, self.open = path, limit, 0, 0, True
        self.tmp = path.with_name(path.name + ".part")
        self.f = open(self.tmp, "wb")

    def add(self, doc_bytes):
        if not self.open:
            return
        need = len(doc_bytes) + (EOT_BYTES if self.docs else 0)
        if self.limit is not None and self.size + need > self.limit:
            self.close()
            return
        if self.docs:
            self.f.write(EOT.encode("utf-8"))
        self.f.write(doc_bytes)
        self.size += need
        self.docs += 1

    def close(self):
        if self.open:
            self.f.close()
            os.replace(self.tmp, self.path)
            self.open = False


def verify(path):
    """Strict UTF-8 check + sha256, streamed."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            decoder.decode(block)
            digest.update(block)
    decoder.decode(b"", final=True)
    return digest.hexdigest()


def update_checksums(data_dir, sums):
    path = data_dir / "CHECKSUMS.sha256"
    table = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            digest, _, name = line.partition("  ")
            table[name] = digest
    table.update(sums)
    path.write_text("".join(f"{table[n]}  {n}\n" for n in sorted(table)), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--sizes", default="10M,100M,1G",
                    type=lambda s: [parse_size(x) for x in s.split(",")])
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--no-full", action="store_true", help="skip <source>.full.txt (saves disk space)")
    args = ap.parse_args()

    raw = args.data_dir / "raw" / args.source
    if not raw.is_dir():
        sys.exit(f"{raw} not found; run scripts/get_data.sh {args.source} first")

    outputs = [Output(args.data_dir / f"{args.source}.{name}.txt", limit) for name, limit in args.sizes]
    if not args.no_full:
        outputs.append(Output(args.data_dir / f"{args.source}.full.txt", None))

    n_docs = 0
    try:
        for doc in SOURCES[args.source](raw):
            if not doc:
                continue
            if args.source != "tinystories" and EOT in doc:
                sys.exit(f"literal {EOT} inside {args.source} document {n_docs}; refusing to insert separators")
            doc_bytes = doc.encode("utf-8")
            for out in outputs:
                out.add(doc_bytes)
            n_docs += 1
            if not any(out.open for out in outputs):
                break
    finally:
        for out in outputs:
            out.close()

    sums = {}
    for out in outputs:
        sums[out.path.name] = verify(out.path)
        print(f"{out.path}: {out.size:,} bytes, {out.docs:,} docs, sha256 {sums[out.path.name][:12]}…")
    update_checksums(args.data_dir, sums)


if __name__ == "__main__":
    main()
