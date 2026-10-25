#!/usr/bin/env bash
# Path B (EAGLE repo, batch 1): what tree SHAPE is best at a fixed budget of 60 verified tokens?
# Every run saves its first trees per turn (--save-trees) for scripts/plot_trees.py.
#
#   EAGLE_DATA=third_party/EAGLE/eagle/data BENCHES=qa bash scripts/run_template_sweep.sh [results/template_sweep]
#
#   1. EAGLE-3 8 x 10, with plain decoding (reference and speedup denominator)
#   2. EAGLE-3 depth x width = 80 trees (1 x 80 ... 40 x 2): realised shapes of EAGLE's own rule
#   3. Fixed templates (option 1): exact widths per depth, profile down / flat / up / diamond at
#      depth 3, 5, 8, 12, scaled to 59 nodes; every node proposes 10 children, all expanded
#   4. Same templates at depth 5 and 8 with only the 3 best nodes per layer expanded (option 3:
#      children per node and nodes expanded per layer set separately)
# A run that fails is reported and skipped. Safe to re-run: finished outputs are skipped.
set -euo pipefail
OUT=${1:-results/template_sweep}
DATA=${EAGLE_DATA:?set EAGLE_DATA to EAGLE/eagle/data}
TEMP=${TEMP:-0}
BENCHES=${BENCHES:-humaneval,gsm8k,qa,sum}
WARMUP=${WARMUP:-1}
SAVE_TREES=${SAVE_TREES:-3}
FAILED=()

run() {  # bench, out-subdir, run_eagle.py tree arguments...
  local B=$1 SUB=$2; shift 2
  python scripts/run_eagle.py --data-dir "$DATA" --bench "$B" --temperature "$TEMP" --budget 60 \
      --max-new-tokens 1024 --warmup "$WARMUP" --save-trees "$SAVE_TREES" --skip-existing "$@" \
      --out-dir "$OUT/$SUB" || { echo "WARNING: $B $SUB failed, continuing" >&2; FAILED+=("$B/$SUB"); }
}

for B in ${BENCHES//,/ }; do
  run "$B" eagle3_s9_k10_b60 --with-ar --steps 9 --topk 10
  for DK in 1:80 2:40 4:20 5:16 10:8 16:5 20:4 40:2; do
    D=${DK%%:*} K=${DK##*:}
    run "$B" "eagle3_s$((D + 1))_k${K}_b60" --steps $((D + 1)) --topk "$K"
  done
  for P in down flat up diamond; do
    for L in 3 5 8 12; do
      run "$B" "tmpl_${P}${L}_b10_eall" --template "shape:$P:$L" --template-label "${P}${L}"
    done
    for L in 5 8; do
      run "$B" "tmpl_${P}${L}_b10_e3" --template "shape:$P:$L" --template-label "${P}${L}" \
          --template-expand 3
    done
  done
done

python scripts/summarize.py "$OUT" --out "$OUT/summary.csv"
echo "done: $OUT/summary.csv"
[ ${#FAILED[@]} -eq 0 ] || echo "failed runs: ${FAILED[*]}"
