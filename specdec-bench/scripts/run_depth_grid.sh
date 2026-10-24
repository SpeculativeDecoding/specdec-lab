#!/usr/bin/env bash
# Path B (EAGLE repo, batch 1): EAGLE-3 vs TALON vs fixed trees of other depths, for choosing
# a depth per workload offline ("Ours"). All questions run; pick each workload's depth on the
# frozen calibration split (specdec.data.split, 30%) and report on the test split (70%).
#
#   EAGLE_DATA=third_party/EAGLE/eagle/data BENCHES=all bash scripts/run_depth_grid.sh [results/depth_grid]
#
# Every tree has width 10 and budget 60 (the TALON paper's EAGLE-3 setting); only the depth
# changes. EAGLE depth = steps - 1. Safe to re-run: finished outputs are skipped.
set -euo pipefail
OUT=${1:-results/depth_grid}
DATA=${EAGLE_DATA:?set EAGLE_DATA to EAGLE/eagle/data}
TEMP=${TEMP:-0}
BENCHES=${BENCHES:-all}
COMMON=(--data-dir "$DATA" --bench "$BENCHES" --temperature "$TEMP" --max-new-tokens 1024 --skip-existing)

# 1) EAGLE-3 reference, depth 8 (steps 9), plus plain decoding on the same model load
python scripts/run_eagle.py "${COMMON[@]}" --with-ar --steps 9 --topk 10 --budget 60 \
    --out-dir "$OUT/eagle3_s9_k10_b60"

# 2) TALON at the paper defaults
python scripts/run_eagle.py "${COMMON[@]}" --adaptive --budget 60 --mu 0.03 --root-k 10 \
    --out-dir "$OUT/adaptive_n60_mu0.03_k10"

# 3) Fixed trees at depth 1, 2, 4, 6, 12 and 20
for STEPS in 2 3 5 7 13 21; do
  python scripts/run_eagle.py "${COMMON[@]}" --steps "$STEPS" --topk 10 --budget 60 \
      --out-dir "$OUT/eagle3_s${STEPS}_k10_b60"
done

python scripts/summarize.py "$OUT" --out "$OUT/summary.csv"
echo "done: $OUT/summary.csv"
