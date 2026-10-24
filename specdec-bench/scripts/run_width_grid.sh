#!/usr/bin/env bash
# Path B (EAGLE repo, batch 1): width variants next to scripts/run_depth_grid.sh, at the same
# budget 60, where width should matter most: deep and narrow (code-like) and shallow and wide
# (QA/summary-like). Plain decoding is rerun on the same model load as the speedup denominator.
#
#   EAGLE_DATA=third_party/EAGLE/eagle/data BENCHES=all bash scripts/run_width_grid.sh [results/width_grid]
#
# EAGLE depth = steps - 1; width = top_k. Safe to re-run: finished outputs are skipped.
set -euo pipefail
OUT=${1:-results/width_grid}
DATA=${EAGLE_DATA:?set EAGLE_DATA to EAGLE/eagle/data}
TEMP=${TEMP:-0}
BENCHES=${BENCHES:-all}
COMMON=(--data-dir "$DATA" --bench "$BENCHES" --temperature "$TEMP" --max-new-tokens 1024 --skip-existing)

AR=(--with-ar)
for TREE in 13:4 21:2 21:4 2:20 3:20 5:16; do   # steps:width = depth 12x4, 20x2, 20x4, 1x20, 2x20, 4x16
  STEPS=${TREE%%:*} K=${TREE##*:}
  python scripts/run_eagle.py "${COMMON[@]}" "${AR[@]}" --steps "$STEPS" --topk "$K" --budget 60 \
      --out-dir "$OUT/eagle3_s${STEPS}_k${K}_b60"
  AR=()   # plain decoding once, with the first tree
done

python scripts/summarize.py "$OUT" --out "$OUT/summary.csv"
echo "done: $OUT/summary.csv"
