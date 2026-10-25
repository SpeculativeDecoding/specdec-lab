#!/usr/bin/env bash
# Local two-GPU port of scripts/run_talon_depthcap.sh (Path B, EAGLE repo, batch 1): TALON with its
# depth capped at EAGLE-3's best depth per workload, and with its root width set to EAGLE-3's best
# width, against EAGLE-3 at budget 60. Every run records the verified tree's mean width at each
# depth (mean_layer_nodes in the .jsonl).
#
#   bash scripts/run_talcon_depthcap_june_a100.sh [results/talon_depthcap_june_a100]
#   GPUS=0 BENCHES=qa bash scripts/run_talcon_depthcap_june_a100.sh   # one GPU, one workload
#
# BEST maps each workload to EAGLE-3's best fixed tree (depth:width) from the depth x width = 80
# sweep on A100 (June's sheet). Per workload:
#   1. EAGLE-3 8 x 10, with plain decoding (the normaliser, as in the sheet)
#   2. EAGLE-3's best D x W, when it is not 8 x 10
#   3. TALON, uncapped, root width 10 (the paper's setting)
#   4. TALON, capped at D + 1 layers (the layers EAGLE-3 depth D drafts), root width 10
#   5. when W != 10: TALON with root width W, uncapped and capped at D + 1 layers
# A run that fails is reported and skipped. Safe to re-run: finished outputs are skipped.
#
# Multi-GPU: the workloads are dealt round-robin over GPUS, and each GPU runs every configuration
# (and plain decoding) for its own workloads, so a speedup's numerator and denominator always come
# from the same GPU and model load. The default order pairs GSM8K with QA and HumanEval with Sum,
# which puts one 6-run workload (QA, Sum: W = 16) on each GPU. Each GPU logs to
# $OUT/logs/gpu<N>.log.
#
# Runs in conda env CONDA_ENV (default eagle; set CONDA_ENV= to use the current environment). The
# EAGLE repo is imported from EAGLE_REPO (default third_party/EAGLE) without installing it.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-results/talon_depthcap_june_a100}
EAGLE_REPO=${EAGLE_REPO:-$PWD/third_party/EAGLE}
DATA=${EAGLE_DATA:-$EAGLE_REPO/eagle/data}
TEMP=${TEMP:-0}
BENCHES=${BENCHES:-gsm8k,humaneval,qa,sum}
BEST=${BEST:-gsm8k:8:10,humaneval:8:10,qa:5:16,sum:5:16}
MU=${MU:-0.03}
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
for B in ${BENCHES//,/ }; do
  tr , '\n' <<<"$BEST" | grep -q "^$B:" || { echo "no BEST entry for $B" >&2; exit 1; }
done

# Download the weights once, before the GPUs start, so two processes never fetch the same files
python - <<'PY'
from huggingface_hub import snapshot_download
from specdec.eagle_runner import DEFAULT_BASE, DEFAULT_EAGLE3
for r in [DEFAULT_BASE, DEFAULT_EAGLE3]:
    print(r, "->", snapshot_download(r, ignore_patterns=["original/*", "*.pth"]), flush=True)
PY

run() {  # bench, out-subdir, run_eagle.py tree arguments...
  local B=$1 SUB=$2; shift 2
  python scripts/run_eagle.py --data-dir "$DATA" --bench "$B" --temperature "$TEMP" \
      --max-new-tokens 1024 --warmup "$WARMUP" --skip-existing "$@" --out-dir "$OUT/$SUB" \
    || { echo "WARNING: $B $SUB failed, continuing" >&2; FAILED+=("$B/$SUB"); }
}

run_gpu() {  # $1 = GPU index, $2 = comma-separated workloads; prints failed runs last
  local GPU=$1 BS=$2 B ENTRY D W K CAP
  FAILED=()
  export CUDA_VISIBLE_DEVICES=$GPU
  echo "GPU $GPU: $BS"
  for B in ${BS//,/ }; do
    ENTRY=$(tr , '\n' <<<"$BEST" | grep "^$B:")
    D=$(cut -d: -f2 <<<"$ENTRY") W=$(cut -d: -f3 <<<"$ENTRY")
    echo "== $B: EAGLE-3 best tree ${D} x ${W}"

    run "$B" eagle3_s9_k10_b60 --with-ar --steps 9 --topk 10 --budget 60
    if [ "$D:$W" != "8:10" ]; then
      run "$B" "eagle3_s$((D + 1))_k${W}_b60" --steps $((D + 1)) --topk "$W" --budget 60
    fi
    # EAGLE depth D drafts D + 1 layers below the root (the context pass adds the first), while
    # TALON's cap counts layers below the root, so cap at D + 1 to match EAGLE-3's tree
    CAP=$((D + 1))
    for K in 10 $([ "$W" != 10 ] && echo "$W"); do
      run "$B" "adaptive_n60_mu${MU}_k${K}" --adaptive --budget 60 --mu "$MU" --root-k "$K"
      run "$B" "adaptive_n60_mu${MU}_k${K}_d${CAP}" --adaptive --budget 60 --mu "$MU" --root-k "$K" \
          --adaptive-max-depth "$CAP"
    done
  done
  echo "GPU $GPU done; failed runs: ${FAILED[*]:-none}"
}

# Deal the workloads round-robin over the GPUs
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
grep -h "done; failed runs" "$OUT"/logs/gpu*.log || true

python scripts/summarize.py "$OUT" --out "$OUT/summary.csv"
echo "done: $OUT/summary.csv"
exit $STATUS
