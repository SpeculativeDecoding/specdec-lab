#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-results/talon_three_bench}
export BENCHES=humaneval,alpaca,mt_bench

exec bash "$SCRIPT_DIR/run_talon_suite.sh" "$OUT"