#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT=$(cd "$HERE/.." && pwd)
WORKSPACE=$(cd "$PROJECT/.." && pwd)

if [ ! -f "$HERE/.env" ]; then
  echo "Create chtc/.env with CHTC_USER and HF_TOKEN first." >&2
  exit 1
fi
set -a
source "$HERE/.env"
set +a
: "${CHTC_USER:?Set CHTC_USER in chtc/.env}"
HOST=${CHTC_HOST:-ap2001.chtc.wisc.edu}
REMOTE="${CHTC_USER}@${HOST}"
ARCHIVE=$(mktemp "${TMPDIR:-/tmp}/specdec-bench.XXXXXX.tar.gz")
trap 'rm -f "$ARCHIVE"' EXIT

tar -czf "$ARCHIVE" \
  --exclude='specdec-bench/.git' \
  --exclude='specdec-bench/chtc/.env' \
  --exclude='specdec-bench/chtc/logs' \
  --exclude='specdec-bench/.venv-*' \
  --exclude='specdec-bench/third_party' \
  --exclude='specdec-bench/results' \
  -C "$WORKSPACE" specdec-bench

ssh "$REMOTE" "mkdir -p ~/specdec-bench/chtc/logs ~/specdec-bench/chtc/results/talon_three_bench"
scp "$ARCHIVE" "$REMOTE:~/specdec-bench/chtc/specdec-bench.tar.gz"
rsync -av --exclude='.env' --exclude='logs/' "$HERE/" "$REMOTE:~/specdec-bench/chtc/"
scp "$HERE/.env" "$REMOTE:~/specdec-bench/chtc/.env"
ssh -t "$REMOTE" "cd ~/specdec-bench/chtc && trap 'rm -f .env specdec-bench.tar.gz' EXIT && condor_submit -spool job_talon.sub"