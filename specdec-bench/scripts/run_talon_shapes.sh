#!/usr/bin/env bash
# Path B (EAGLE repo, batch 1): TALON (mu 0.03, K 10, budget 60) with every tree's shape recorded
# ("tree_shapes" in each turn: depth, widest layer, most nodes expanded in one layer, most children
# of one node). scripts/analyze_talon_shapes.py then counts how many TALON trees fit SGLang's
# fixed tree (9 steps, top-k 10), i.e. how much of TALON TALON-S can express.
#
#   EAGLE_DATA=third_party/EAGLE/eagle/data BENCHES=humaneval,gsm8k,mt_bench bash scripts/run_talon_shapes.sh [results/talon_shapes]
set -euo pipefail
OUT=${1:-results/talon_shapes}
DATA=${EAGLE_DATA:?set EAGLE_DATA to EAGLE/eagle/data}
TEMP=${TEMP:-0}
BENCHES=${BENCHES:-all}
WARMUP=${WARMUP:-3}
COMMON=(--data-dir "$DATA" --bench "$BENCHES" --temperature "$TEMP" --max-new-tokens 1024
        --warmup "$WARMUP" --skip-existing)

python scripts/run_eagle.py "${COMMON[@]}" --adaptive --budget 60 --mu 0.03 --root-k 10 --save-trees 3 \
    --out-dir "$OUT/adaptive_n60_mu0.03_k10"

python scripts/summarize.py "$OUT" --out "$OUT/summary.csv"
python scripts/analyze_talon_shapes.py "$OUT" --out "$OUT/shapes.csv"
echo "done: $OUT/summary.csv $OUT/shapes.csv"
