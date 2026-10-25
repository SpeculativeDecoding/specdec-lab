#!/usr/bin/env bash
# Path B (EAGLE repo, batch 1): TALON with its depth capped at EAGLE-3's best depth per workload,
# and with its root width set to EAGLE-3's best width, against EAGLE-3 at budget 60. Every run
# records the verified tree's mean width at each depth (mean_layer_nodes in the .jsonl).
#
#   EAGLE_DATA=third_party/EAGLE/eagle/data BENCHES=qa bash scripts/run_talon_depthcap.sh [results/talon_depthcap]
#
# BEST maps each workload to EAGLE-3's best fixed tree (depth:width) from the depth x width = 80
# sweep on A100 (June's sheet). Per workload:
#   1. EAGLE-3 8 x 10, with plain decoding (the normaliser, as in the sheet)
#   2. EAGLE-3's best D x W, when it is not 8 x 10
#   3. TALON, uncapped, root width 10 (the paper's setting)
#   4. TALON, depth capped at D, root width 10
#   5. when W != 10: TALON with root width W, uncapped and capped at D
# A run that fails is reported and skipped. Safe to re-run: finished outputs are skipped.
set -euo pipefail
OUT=${1:-results/talon_depthcap}
DATA=${EAGLE_DATA:?set EAGLE_DATA to EAGLE/eagle/data}
TEMP=${TEMP:-0}
BENCHES=${BENCHES:-gsm8k,humaneval,qa,sum}
BEST=${BEST:-gsm8k:8:10,humaneval:8:10,qa:5:16,sum:5:16}
MU=${MU:-0.03}
WARMUP=${WARMUP:-1}
FAILED=()

run() {  # bench, out-subdir, run_eagle.py tree arguments...
  local B=$1 SUB=$2; shift 2
  python scripts/run_eagle.py --data-dir "$DATA" --bench "$B" --temperature "$TEMP" \
      --max-new-tokens 1024 --warmup "$WARMUP" --skip-existing "$@" --out-dir "$OUT/$SUB" \
    || { echo "WARNING: $B $SUB failed, continuing" >&2; FAILED+=("$B/$SUB"); }
}

for B in ${BENCHES//,/ }; do
  ENTRY=$(tr , '\n' <<<"$BEST" | grep "^$B:" || true)
  [ -n "$ENTRY" ] || { echo "no BEST entry for $B" >&2; exit 1; }
  D=$(cut -d: -f2 <<<"$ENTRY") W=$(cut -d: -f3 <<<"$ENTRY")
  echo "== $B: EAGLE-3 best tree ${D} x ${W}"

  run "$B" eagle3_s9_k10_b60 --with-ar --steps 9 --topk 10 --budget 60
  if [ "$D:$W" != "8:10" ]; then
    run "$B" "eagle3_s$((D + 1))_k${W}_b60" --steps $((D + 1)) --topk "$W" --budget 60
  fi
  for K in 10 $([ "$W" != 10 ] && echo "$W"); do
    run "$B" "adaptive_n60_mu${MU}_k${K}" --adaptive --budget 60 --mu "$MU" --root-k "$K"
    run "$B" "adaptive_n60_mu${MU}_k${K}_d${D}" --adaptive --budget 60 --mu "$MU" --root-k "$K" \
        --adaptive-max-depth "$D"
  done
done

python scripts/summarize.py "$OUT" --out "$OUT/summary.csv"
echo "done: $OUT/summary.csv"
[ ${#FAILED[@]} -eq 0 ] || echo "failed runs: ${FAILED[*]}"
