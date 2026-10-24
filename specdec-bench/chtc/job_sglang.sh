#!/usr/bin/env bash
# Path A on CHTC: run scripts/run_sglang_sweep.py inside the official SGLang container.
# Arguments (set in job_sglang.sub): config file, comma-separated benchmarks, results folder name.
set -euo pipefail

source .env
: "${HF_TOKEN:?Set HF_TOKEN in .env with access to Llama-3.1-8B-Instruct}"

CONFIG=${1:-configs/smoke.yaml}
BENCHES=${2:-}
RUN=${3:-sglang_smoke}
EAGLE_COMMIT=cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b   # same pin as envs/setup_eagle_env.sh

SCRATCH=${_CONDOR_SCRATCH_DIR:-$PWD}
SUBMIT_DIR=$PWD
OUTPUT_DIR="$SUBMIT_DIR/results/$RUN"

export HOME="$SCRATCH/home"
export HF_HOME="$SCRATCH/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN
mkdir -p "$HOME" "$HF_HOME" "$SCRATCH/work" "$OUTPUT_DIR"

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

# Question files come from the EAGLE repo; a GitHub archive of the pinned commit needs no git.
echo "Fetching EAGLE question files @ $EAGLE_COMMIT"
mkdir -p third_party
python3 - "$EAGLE_COMMIT" <<'PY'
import io, sys, tarfile, urllib.request
commit = sys.argv[1]
url = f"https://github.com/SafeAILab/EAGLE/archive/{commit}.tar.gz"
data = urllib.request.urlopen(url, timeout=300).read()
with tarfile.open(fileobj=io.BytesIO(data)) as t:
    try:
        t.extractall("third_party", filter="data")
    except TypeError:                     # Python without extraction filters
        t.extractall("third_party")
PY
mv "third_party/EAGLE-$EAGLE_COMMIT" third_party/EAGLE
DATA="$PWD/third_party/EAGLE/eagle/data"

python3 -c "import yaml" 2>/dev/null || python3 -m pip install -q --user pyyaml

# Job config: optionally restrict to some benchmarks, and use a free port so two jobs on one
# machine cannot collide on the default 30000. Every other setting is kept.
python3 - "$CONFIG" "$BENCHES" > "$SCRATCH/sweep_config.yaml" <<'PY'
import socket, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
if sys.argv[2]:
    cfg.setdefault("run", {})["benches"] = sys.argv[2].split(",")
s = socket.socket(); s.bind(("127.0.0.1", 0))
cfg.setdefault("server", {})["port"] = s.getsockname()[1]
s.close()
yaml.safe_dump(cfg, sys.stdout, sort_keys=False)
PY
CONFIG="$SCRATCH/sweep_config.yaml"
grep -A1 "port:" "$CONFIG" | head -1

# Download the weights before any server starts, so no launch waits on a 16 GB download
echo "Downloading models"
python3 - <<'PY'
from huggingface_hub import snapshot_download
for r in ["meta-llama/Llama-3.1-8B-Instruct", "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"]:
    print(r, "->", snapshot_download(r, ignore_patterns=["original/*", "*.pth"]), flush=True)
PY

# SGLang's EAGLE-3 draft declares 2048 positions and Llama-3.1 declares 131072; v0.5.22 refuses the
# mismatch unless overridden. (Path B's EAGLE runs use a 4096-token KV cache, run_eagle.py --max-length.)
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Register specdec's SGLang plugin (TALON-S, active only for mode: talon servers) without pip or
# root: a minimal dist-info with the entry point, plus the repo, on PYTHONPATH. SGLang loads
# plugins from group sglang.srt.plugins in every process, including the scheduler.
PLUGIN_DIR="$SCRATCH/pylib/specdec_bench-0.1.0.dist-info"
mkdir -p "$PLUGIN_DIR"
printf 'Metadata-Version: 2.1\nName: specdec-bench\nVersion: 0.1.0\n' > "$PLUGIN_DIR/METADATA"
printf '[sglang.srt.plugins]\ntalon = specdec.sglang_talon_plugin:plugin\n' > "$PLUGIN_DIR/entry_points.txt"
export PYTHONPATH="$PWD:$SCRATCH/pylib${PYTHONPATH:+:$PYTHONPATH}"
python3 -c "from importlib.metadata import entry_points as e; import specdec.sglang_talon_plugin; print('sglang plugins:', [x.value for x in e(group='sglang.srt.plugins')])"

echo "GPU:"
nvidia-smi
python3 -c "import sglang, torch; print('sglang', sglang.__version__, '| torch', torch.__version__)"
echo "Starting SGLang sweep: $CONFIG (benches: ${BENCHES:-from config}) -> results/$RUN"
python3 scripts/run_sglang_sweep.py --config "$CONFIG" --data-dir "$DATA" \
	--out-dir "results/$RUN" --skip-existing
echo "done: results/$RUN"
