#!/usr/bin/env bash
# Path B (EAGLE repo, batch 1): EAGLE-3 fixed trees trading depth for width at a constant
# depth x width = PRODUCT (80 = EAGLE-3's 8 x 10), all at budget 60, with plain decoding on the
# same model load as the speedup denominator. One warm-up per timed run.
#
#   EAGLE_DATA=third_party/EAGLE/eagle/data BENCHES=humaneval bash scripts/run_depth_sweep.sh [results/depth_sweep]
#
# DEPTHS are EAGLE depths (steps = depth + 1); width = PRODUCT / depth, so each depth must divide
# PRODUCT. Depth 80 x width 1 is left out: a 1-wide chain holds only `depth` drafted tokens, so at
# budget 60 it is the same tree as a 59-deep chain. Ascending, so the slowest (deepest) trees run
# last. A tree that fails is reported and skipped. Safe to re-run: finished outputs are skipped.
set -euo pipefail
OUT=${1:-results/depth_sweep}
DATA=${EAGLE_DATA:?set EAGLE_DATA to EAGLE/eagle/data}
TEMP=${TEMP:-0}
BENCHES=${BENCHES:-humaneval,sum,qa}
DEPTHS=${DEPTHS:-1 2 4 5 8 10 16 20 40}
PRODUCT=${PRODUCT:-80}
WARMUP=${WARMUP:-1}
COMMON=(--data-dir "$DATA" --bench "$BENCHES" --temperature "$TEMP" --max-new-tokens 1024
        --warmup "$WARMUP" --skip-existing)

run_tree() {  # $1 = depth, rest = extra run_eagle.py arguments
  local D=$1; shift
  if (( PRODUCT % D )); then echo "depth $D does not divide PRODUCT=$PRODUCT" >&2; return 1; fi
  local STEPS=$((D + 1)) K=$((PRODUCT / D))
  python scripts/run_eagle.py "${COMMON[@]}" "$@" --steps "$STEPS" --topk "$K" --budget 60 \
      --out-dir "$OUT/eagle3_s${STEPS}_k${K}_b60"
}

# EAGLE-3's own 8 x 10 first, with plain decoding: the known-good reference and the denominator
run_tree 8 --with-ar
FAILED=()
for D in $DEPTHS; do
  [ "$D" = 8 ] && continue
  run_tree "$D" || { echo "WARNING: depth $D x width $((PRODUCT / D)) failed, continuing" >&2; FAILED+=("$D"); }
done

python scripts/summarize.py "$OUT" --out "$OUT/summary.csv"
echo "done: $OUT/summary.csv"
[ ${#FAILED[@]} -eq 0 ] || echo "failed depths: ${FAILED[*]}"
