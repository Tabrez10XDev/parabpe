#!/usr/bin/env bash
# S8: builds the temporary corpora, runs every benchmark on mains power and plots the results.
# Run bench/configs/b4_sweep.yaml and set the defaults first: batch.yaml and scale.yaml use them.
# Stops at the first failure or if the Mac leaves mains power. Rerunning appends to results/*.csv,
# so move old CSVs away first. Afterwards delete the corpora with: scripts/run_s8.sh --clean
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -I"
X6=data/wikitext.x6.txt
WEAK="50M,100M,150M,200M,250M,300M,400M,550M"

if [[ "${1:-}" == "--clean" ]]; then
  rm -f data/tinystories.full.txt "$X6"
  for size in 50M 150M 200M 250M 300M 400M 550M; do rm -f "data/tinystories.$size.txt"; done
  exit 0
fi

[[ -f data/tinystories.full.txt ]] || $PY scripts/slice_data.py --source tinystories --sizes "$WEAK"
if [[ ! -f "$X6" ]]; then
  for i in 1 2 3 4 5 6; do
    cat data/wikitext.full.txt
    [[ $i == 6 ]] || printf '<|endoftext|>'
  done > "$X6"
fi

for config in batch owners owner_load span main threshold reduce naive weak scale; do
  $PY bench/run_bench.py "bench/configs/$config.yaml" --require-ac
done
mkdir -p results
$PY scripts/profile_phases.py --small data/tinystories.10M.txt --large data/tinystories.100M.txt \
  --threads 11 --out results/profile_tinystories.md
$PY scripts/profile_phases.py --small data/wikitext.10M.txt --large data/wikitext.100M.txt \
  --threads 11 --out results/profile_wikitext.md
$PY bench/plot.py results/*.csv --out-dir results/plots
