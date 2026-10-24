#!/usr/bin/env bash
set -euo pipefail

source .env
: "${CHTC_USER:?Set CHTC_USER in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env with access to Llama-3.1-8B-Instruct}"

# Arguments (set in job_adaptive.sub): comma-separated benchmarks, the results folder name and
# the Path B script to run (scripts/run_adaptive_suite.sh or scripts/run_depth_grid.sh)
BENCHES=${1:-humaneval,alpaca,mt_bench}
RUN=${2:-adaptive_three_bench}
SUITE=${3:-scripts/run_adaptive_suite.sh}

SCRATCH=${_CONDOR_SCRATCH_DIR:-$PWD}
SUBMIT_DIR=$PWD
OUTPUT_DIR="$SUBMIT_DIR/results/$RUN"

export HOME="$SCRATCH/home"
export HF_HOME="$SCRATCH/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN
mkdir -p "$HOME" "$HF_HOME" "$SCRATCH/work"
mkdir -p "$OUTPUT_DIR"

heartbeat() {
	while true; do
		printf '[heartbeat] %s CHTC job is alive\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
		sleep 300
	done
}

heartbeat &
HEARTBEAT_PID=$!

copy_results() {
	local source_dir="$SCRATCH/work/specdec-bench/results/$RUN"
	if [ -d "$source_dir" ]; then
		mkdir -p "$OUTPUT_DIR"
		cp -a "$source_dir/." "$OUTPUT_DIR/"
	fi
}

finish_job() {
	local job_status=$?
	trap - EXIT
	kill "$HEARTBEAT_PID" 2>/dev/null || true
	wait "$HEARTBEAT_PID" 2>/dev/null || true
	copy_results || echo "WARNING: failed to copy benchmark results to Condor output directory" >&2
	exit "$job_status"
}

trap finish_job EXIT

cd "$SCRATCH/work"

echo "Extracting transferred project archive"
cp "$SUBMIT_DIR/specdec-bench.tar.gz" .
tar -xzf specdec-bench.tar.gz
rm specdec-bench.tar.gz
cd specdec-bench

if ! command -v git >/dev/null 2>&1; then
	echo "Git is missing from the container; installing it with conda (no root needed)"
	conda create -y -q -p "$SCRATCH/gitenv" -c conda-forge git
	export PATH="$SCRATCH/gitenv/bin:$PATH"
	git --version
fi

echo "Setting up the pinned EAGLE environment"
bash envs/setup_eagle_env.sh
source .venv-eagle/bin/activate
export EAGLE_DATA="$PWD/third_party/EAGLE/eagle/data"

mkdir -p "results/$RUN"

echo "GPU:"
nvidia-smi
echo "Starting $SUITE on $BENCHES -> results/$RUN"
TEMP=0 BENCHES="$BENCHES" bash "$SUITE" "results/$RUN"
