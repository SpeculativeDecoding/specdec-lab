#!/usr/bin/env bash
# Path B (EAGLE repo, batch 1): EAGLE-3 fixed trees at many depths, width 10, budget 60, with
# plain decoding on the same model load as the speedup denominator. One warm-up per timed run.
#
#   EAGLE_DATA=third_party/EAGLE/eagle/data BENCHES=humaneval bash scripts/run_depth_sweep.sh [results/depth_sweep]
#
# DEPTHS are EAGLE depths (steps = depth + 1). Ascending, so the slowest (deepest) trees run
# last and everything before them is saved if one fails. Safe to re-run: finished outputs are skipped.
set -euo pipefail
OUT=${1:-results/depth_sweep}
DATA=${EAGLE_DATA:?set EAGLE_DATA to EAGLE/eagle/data}
TEMP=${TEMP:-0}
BENCHES=${BENCHES:-humaneval,sum,qa}
DEPTHS=${DEPTHS:-1 2 4 5 8 10 16 20 40 80}
WARMUP=${WARMUP:-1}
COMMON=(--data-dir "$DATA" --bench "$BENCHES" --temperature "$TEMP" --max-new-tokens 1024
        --warmup "$WARMUP" --skip-existing)

AR=(--with-ar)
for D in $DEPTHS; do
  STEPS=$((D + 1))
  python scripts/run_eagle.py "${COMMON[@]}" "${AR[@]}" --steps "$STEPS" --topk 10 --budget 60 \
      --out-dir "$OUT/eagle3_s${STEPS}_k10_b60"
  AR=()   # plain decoding once, with the first tree
done

python scripts/summarize.py "$OUT" --out "$OUT/summary.csv"
echo "done: $OUT/summary.csv"
