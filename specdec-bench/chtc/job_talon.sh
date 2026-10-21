#!/usr/bin/env bash
set -euo pipefail

source .env
: "${CHTC_USER:?Set CHTC_USER in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env with access to Llama-3.1-8B-Instruct}"

SCRATCH=${_CONDOR_SCRATCH_DIR:-$PWD}
SUBMIT_DIR=$PWD

export HOME="$SCRATCH/home"
export HF_HOME="$SCRATCH/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN
mkdir -p "$HOME" "$HF_HOME" "$SCRATCH/work"
cd "$SCRATCH/work"

echo "Extracting transferred project archive"
cp "$SUBMIT_DIR/specdec-bench.tar.gz" .
tar -xzf specdec-bench.tar.gz
rm specdec-bench.tar.gz
cd specdec-bench

echo "Setting up the pinned EAGLE environment"
bash envs/setup_eagle_env.sh
source .venv-eagle/bin/activate
export EAGLE_DATA="$PWD/third_party/EAGLE/eagle/data"

mkdir -p results/talon_three_bench

echo "GPU:"
nvidia-smi
echo "Starting TALON suite"
bash scripts/run_talon_suite.sh results/talon_three_bench