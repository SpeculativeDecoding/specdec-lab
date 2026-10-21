#!/usr/bin/env bash
# Reproduce adaptive tree's Llama-3.1-8B setting and budget/mu ablations across the
# selected benchmarks. Defaults to all available benchmarks.
#
#   EAGLE_DATA=third_party/EAGLE/eagle/data bash scripts/run_adaptive_suite.sh [results/adaptive]
#   BENCHES=humaneval,alpaca,mt_bench bash scripts/run_adaptive_suite.sh [results/custom]
#
# Safe to re-run after an interruption: finished outputs are skipped.
# Every run uses the same draft weights (yuhuili/EAGLE3-LLaMA3.1-Instruct-8B), T=0, 1024 new tokens.
set -euo pipefail
OUT=${1:-results/adaptive}
DATA=${EAGLE_DATA:?set EAGLE_DATA to EAGLE/eagle/data}
TEMP=${TEMP:-0}
BENCHES=${BENCHES:-all}
COMMON=(--data-dir "$DATA" --bench "$BENCHES" --temperature "$TEMP" --max-new-tokens 1024 --skip-existing)

# 1) EAGLE-3 baseline as in the paper (K=10, D=8, N=60 -> steps 9) + plain decoding, one load
python scripts/run_eagle.py "${COMMON[@]}" --with-ar --steps 9 --topk 10 --budget 60 \
    --out-dir "$OUT/eagle3_s9_k10_b60"

# 2) adaptive tree at the paper defaults
python scripts/run_eagle.py "${COMMON[@]}" --adaptive --budget 60 --mu 0.03 --root-k 10 \
    --out-dir "$OUT/adaptive_n60_mu0.03_k10"

# 3) Budget ablation (paper Fig. 12): adaptive tree and EAGLE-3 at N = 32, 48, 96
for N in 32 48 96; do
  python scripts/run_eagle.py "${COMMON[@]}" --adaptive --budget "$N" --mu 0.03 --root-k 10 \
      --out-dir "$OUT/adaptive_n${N}_mu0.03_k10"
  python scripts/run_eagle.py "${COMMON[@]}" --steps 9 --topk 10 --budget "$N" \
      --out-dir "$OUT/eagle3_s9_k10_b${N}"
done

# 4) Threshold ablation (paper Fig. 13): mu = 0.01 ... 0.05 at N = 60
for MU in 0.01 0.02 0.04 0.05; do
  python scripts/run_eagle.py "${COMMON[@]}" --adaptive --budget 60 --mu "$MU" --root-k 10 \
      --out-dir "$OUT/adaptive_n60_mu${MU}_k10"
done

python scripts/summarize.py "$OUT" --out "$OUT/summary.csv"
echo "done: $OUT/summary.csv"
