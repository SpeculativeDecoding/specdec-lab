#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-results/adaptive_three_bench}
export BENCHES=humaneval,alpaca,mt_bench

exec bash "$SCRIPT_DIR/run_adaptive_suite.sh" "$OUT"