#!/usr/bin/env bash
# Downloads raw corpora into data/raw/<name>/ (one fresh directory per source).
# Usage: scripts/get_data.sh [tinystories|wikitext|enwik9|all]   (default: tinystories)
# Downloaded files are untrusted data: nothing in data/ is ever executed.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RAW="$ROOT/data/raw"
HF=https://huggingface.co/datasets

fetch() {  # fetch <url> <dest>: skip if present, download to a .part file, then rename
  local url=$1 dest=$2
  if [[ -f "$dest" ]]; then
    echo "have $dest"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  echo "get  $url"
  curl -L --fail --retry 3 -C - -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

get_tinystories() {
  fetch "$HF/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" \
    "$RAW/tinystories/TinyStoriesV2-GPT4-train.txt"
}

get_wikitext() {
  for shard in 00000 00001; do
    fetch "$HF/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-${shard}-of-00002.parquet" \
      "$RAW/wikitext/train-${shard}-of-00002.parquet"
  done
}

get_enwik9() {
  fetch "http://mattmahoney.net/dc/enwik9.zip" "$RAW/enwik9/enwik9.zip"
  if [[ ! -f "$RAW/enwik9/enwik9" ]]; then
    unzip -o -d "$RAW/enwik9" "$RAW/enwik9/enwik9.zip" enwik9
  fi
}

case "${1:-tinystories}" in
  tinystories) get_tinystories ;;
  wikitext) get_wikitext ;;
  enwik9) get_enwik9 ;;
  all) get_tinystories; get_wikitext; get_enwik9 ;;
  *) echo "usage: $0 [tinystories|wikitext|enwik9|all]" >&2; exit 2 ;;
esac
