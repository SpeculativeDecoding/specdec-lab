#!/usr/bin/env bash
set -euo pipefail

source .env
: "${CHTC_USER:?Set CHTC_USER in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env with access to Llama-3.1-8B-Instruct}"

SCRATCH=${_CONDOR_SCRATCH_DIR:-$PWD}
STAGING="/staging/${CHTC_USER}"
ARCHIVE="${STAGING}/specdec-bench.tar.gz"
SAVED_RESULTS="${STAGING}/specdec-talon-three-bench-results"

export HOME="$SCRATCH/home"
export HF_HOME="$SCRATCH/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN
mkdir -p "$HOME" "$HF_HOME" "$SCRATCH/work"
cd "$SCRATCH/work"

echo "Fetching project archive from CHTC staging"
cp "$ARCHIVE" .
tar -xzf specdec-bench.tar.gz
rm specdec-bench.tar.gz
cd specdec-bench

echo "Setting up the pinned EAGLE environment"
bash envs/setup_eagle_env.sh
source .venv-eagle/bin/activate
export EAGLE_DATA="$PWD/third_party/EAGLE/eagle/data"

mkdir -p results/talon_three_bench "$SAVED_RESULTS"
if [ -d "$SAVED_RESULTS" ]; then
  cp -a "$SAVED_RESULTS/." results/talon_three_bench/
fi

sync_results() {
  cp -a results/talon_three_bench/. "$SAVED_RESULTS/"
}

periodic_sync() {
  while true; do
    sleep 300
    sync_results
  done
}

periodic_sync &
SYNC_PID=$!
finish() {
  kill "$SYNC_PID" 2>/dev/null || true
  sync_results
}
trap finish EXIT

echo "GPU:"
nvidia-smi
echo "Starting resumable TALON suite"
bash scripts/run_talon_suite.sh results/talon_three_bench