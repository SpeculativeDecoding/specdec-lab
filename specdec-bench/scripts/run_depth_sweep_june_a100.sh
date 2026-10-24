#!/usr/bin/env bash
# Path B (EAGLE repo, batch 1): EAGLE-3 fixed trees trading depth for width at a constant
# depth x width = PRODUCT (80 = EAGLE-3's 8 x 10), all at budget 60, with plain decoding on the
# same model load as the speedup denominator. One warm-up per timed run.
#
#   bash scripts/run_depth_sweep.sh [results/depth-sweep-a100]
#   GPUS=0 BENCHES=humaneval bash scripts/run_depth_sweep.sh results/depth_sweep   # one GPU
#
# DEPTHS are EAGLE depths (steps = depth + 1); width = PRODUCT / depth, so each depth must divide
# PRODUCT. Depth 80 x width 1 is left out: a 1-wide chain holds only `depth` drafted tokens, so at
# budget 60 it is the same tree as a 59-deep chain. Ascending, so the slowest (deepest) trees run
# last. A tree that fails is reported and skipped. Safe to re-run: finished outputs are skipped.
#
# Multi-GPU: the benchmarks are dealt round-robin over GPUS, and each GPU runs every depth (and
# plain decoding) for its own benchmarks, so a speedup's numerator and denominator always come from
# the same GPU and model load. The default order pairs HumanEval with QA and GSM8K with Sum, which
# roughly balances output tokens. Each GPU logs to $OUT/logs/gpu<N>.log.
#
# Runs in conda env CONDA_ENV (default eagle; set CONDA_ENV= to use the current environment). The
# EAGLE repo is imported from EAGLE_REPO (default third_party/EAGLE) without installing it.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-results/depth-sweep-a100}
EAGLE_REPO=${EAGLE_REPO:-$PWD/third_party/EAGLE}
DATA=${EAGLE_DATA:-$EAGLE_REPO/eagle/data}
TEMP=${TEMP:-0}
BENCHES=${BENCHES:-humaneval,gsm8k,qa,sum}
DEPTHS=${DEPTHS:-1 2 4 5 8 10 16 20 40}
PRODUCT=${PRODUCT:-80}
WARMUP=${WARMUP:-1}
GPUS=${GPUS:-0 1}
CONDA_ENV=${CONDA_ENV-eagle}

if [ -n "$CONDA_ENV" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi
[ -d "$DATA" ] || { echo "no question files at $DATA (set EAGLE_DATA)" >&2; exit 1; }
export PYTHONPATH="$EAGLE_REPO${PYTHONPATH:+:$PYTHONPATH}"
python -c "import eagle.model.ea_model" || { echo "cannot import EAGLE from $EAGLE_REPO" >&2; exit 1; }
for D in $DEPTHS; do
  if (( PRODUCT % D )); then echo "depth $D does not divide PRODUCT=$PRODUCT" >&2; exit 1; fi
done

# Download the weights once, before the GPUs start, so two processes never fetch the same files
python - <<'PY'
from huggingface_hub import snapshot_download
from specdec.eagle_runner import DEFAULT_BASE, DEFAULT_EAGLE3
for r in [DEFAULT_BASE, DEFAULT_EAGLE3]:
    print(r, "->", snapshot_download(r, ignore_patterns=["original/*", "*.pth"]), flush=True)
PY

run_tree() {  # $1 = depth, $2 = benchmarks, rest = extra run_eagle.py arguments
  local D=$1 B=$2; shift 2
  local STEPS=$((D + 1)) K=$((PRODUCT / D))
  python scripts/run_eagle.py --data-dir "$DATA" --bench "$B" --temperature "$TEMP" \
      --max-new-tokens 1024 --warmup "$WARMUP" --skip-existing \
      "$@" --steps "$STEPS" --topk "$K" --budget 60 --out-dir "$OUT/eagle3_s${STEPS}_k${K}_b60"
}

run_gpu() {  # $1 = GPU index, $2 = comma-separated benchmarks; prints failed depths last
  local GPU=$1 B=$2 FAILED=()
  export CUDA_VISIBLE_DEVICES=$GPU
  echo "GPU $GPU: $B"
  # EAGLE-3's own 8 x 10 first, with plain decoding: the known-good reference and the denominator
  run_tree 8 "$B" --with-ar || FAILED+=(8)
  for D in $DEPTHS; do
    [ "$D" = 8 ] && continue
    run_tree "$D" "$B" || { echo "WARNING: depth $D x width $((PRODUCT / D)) failed on $B, continuing" >&2; FAILED+=("$D"); }
  done
  echo "GPU $GPU done; failed depths: ${FAILED[*]:-none}"
}

# Deal the benchmarks round-robin over the GPUs
read -ra GPU_LIST <<< "$GPUS"
IFS=, read -ra BENCH_LIST <<< "$BENCHES"
declare -A LANE
for i in "${!BENCH_LIST[@]}"; do
  G=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
  LANE[$G]=${LANE[$G]:+${LANE[$G]},}${BENCH_LIST[$i]}
done

mkdir -p "$OUT/logs"
PIDS=()
for G in "${GPU_LIST[@]}"; do
  [ -n "${LANE[$G]:-}" ] || continue
  run_gpu "$G" "${LANE[$G]}" > "$OUT/logs/gpu$G.log" 2>&1 &
  PIDS+=($!)
  echo "GPU $G: ${LANE[$G]} -> $OUT/logs/gpu$G.log"
done
STATUS=0
for P in "${PIDS[@]}"; do wait "$P" || STATUS=1; done
grep -h "done; failed depths" "$OUT"/logs/gpu*.log || true

python scripts/summarize.py "$OUT" --out "$OUT/summary.csv"
echo "done: $OUT/summary.csv"
exit $STATUS
