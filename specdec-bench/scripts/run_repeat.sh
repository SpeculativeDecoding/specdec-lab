#!/usr/bin/env bash
# Path B (EAGLE repo, batch 1): one more repeat of the headline configurations, to measure
# run-to-run variance (the proposal's success criterion: a gain larger than the variance
# across three repeats). Same settings as scripts/run_depth_grid.sh, including 3 warm-ups.
#
#   EAGLE_DATA=third_party/EAGLE/eagle/data BENCHES=humaneval,gsm8k,mt_bench bash scripts/run_repeat.sh [results/repeat2]
#
# Runs, per workload: EAGLE-3 8 x 10 with plain decoding, TALON (mu 0.03, K 10), and the fixed
# trees "Ours" picks or ties with: 6 x 10 and 5 x 16. All at budget 60, every question.
set -euo pipefail
OUT=${1:-results/repeat}
DATA=${EAGLE_DATA:?set EAGLE_DATA to EAGLE/eagle/data}
TEMP=${TEMP:-0}
BENCHES=${BENCHES:-all}
WARMUP=${WARMUP:-3}
COMMON=(--data-dir "$DATA" --bench "$BENCHES" --temperature "$TEMP" --max-new-tokens 1024
        --warmup "$WARMUP" --skip-existing)

python scripts/run_eagle.py "${COMMON[@]}" --with-ar --steps 9 --topk 10 --budget 60 --out-dir "$OUT/eagle3_s9_k10_b60"
python scripts/run_eagle.py "${COMMON[@]}" --adaptive --budget 60 --mu 0.03 --root-k 10 --out-dir "$OUT/adaptive_n60_mu0.03_k10"
python scripts/run_eagle.py "${COMMON[@]}" --steps 7 --topk 10 --budget 60 --out-dir "$OUT/eagle3_s7_k10_b60"
python scripts/run_eagle.py "${COMMON[@]}" --steps 6 --topk 16 --budget 60 --out-dir "$OUT/eagle3_s6_k16_b60"

python scripts/summarize.py "$OUT" --out "$OUT/summary.csv"
echo "done: $OUT/summary.csv"
