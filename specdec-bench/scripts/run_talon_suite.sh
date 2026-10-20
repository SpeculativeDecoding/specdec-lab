#!/usr/bin/env bash
# Reproduce TALON's Llama-3.1-8B setting (Table 1 row, Fig. 12 budget and Fig. 13 mu ablations)
# on the EAGLE-repo path (batch size 1), then summarize.
#
#   EAGLE_DATA=third_party/EAGLE/eagle/data bash scripts/run_talon_suite.sh [results/talon]
#
# Safe to re-run after an interruption: finished outputs are skipped.
# Every run uses the same draft weights (yuhuili/EAGLE3-LLaMA3.1-Instruct-8B), T=0, 1024 new tokens.
set -euo pipefail
OUT=${1:-results/talon}
DATA=${EAGLE_DATA:?set EAGLE_DATA to EAGLE/eagle/data}
TEMP=${TEMP:-0}
COMMON=(--data-dir "$DATA" --bench all --temperature "$TEMP" --max-new-tokens 1024 --skip-existing)

# 1) EAGLE-3 baseline as in the paper (K=10, D=8, N=60 -> steps 9) + plain decoding, one load
python scripts/run_eagle.py "${COMMON[@]}" --with-ar --steps 9 --topk 10 --budget 60 \
    --out-dir "$OUT/eagle3_s9_k10_b60"

# 2) TALON at the paper defaults
python scripts/run_eagle.py "${COMMON[@]}" --talon --budget 60 --mu 0.03 --root-k 10 \
    --out-dir "$OUT/talon_n60_mu0.03_k10"

# 3) Budget ablation (paper Fig. 12): TALON and EAGLE-3 at N = 32, 48, 96
for N in 32 48 96; do
  python scripts/run_eagle.py "${COMMON[@]}" --talon --budget "$N" --mu 0.03 --root-k 10 \
      --out-dir "$OUT/talon_n${N}_mu0.03_k10"
  python scripts/run_eagle.py "${COMMON[@]}" --steps 9 --topk 10 --budget "$N" \
      --out-dir "$OUT/eagle3_s9_k10_b${N}"
done

# 4) Threshold ablation (paper Fig. 13): mu = 0.01 ... 0.05 at N = 60
for MU in 0.01 0.02 0.04 0.05; do
  python scripts/run_eagle.py "${COMMON[@]}" --talon --budget 60 --mu "$MU" --root-k 10 \
      --out-dir "$OUT/talon_n60_mu${MU}_k10"
done

python scripts/summarize.py "$OUT" --out "$OUT/summary.csv"
echo "done: $OUT/summary.csv"
