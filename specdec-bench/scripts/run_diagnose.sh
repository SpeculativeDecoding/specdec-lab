#!/usr/bin/env bash
# Classify every speculative-vs-plain mismatch as an fp16 near-tie or not, with
# scripts/diagnose_mismatch.py (one target-model forward pass per differing answer).
#
#   EAGLE_DATA=third_party/EAGLE/eagle/data bash scripts/run_diagnose.sh [results/diagnose]
#
# Inputs: diag_inputs/<bench>/<method>.jsonl and diag_inputs/<bench>/ar.jsonl (copied from a
# results folder before upload, because the CHTC upload leaves results/ out). Writes one report
# per (bench, method) and a combined summary.txt.
set -uo pipefail
OUT=${1:-results/diagnose}
DATA=${EAGLE_DATA:?set EAGLE_DATA to EAGLE/eagle/data}
IN=${DIAG_INPUTS:-diag_inputs}
mkdir -p "$OUT"
: > "$OUT/summary.txt"
for dir in "$IN"/*/; do
  B=$(basename "$dir")
  for f in "$dir"*.jsonl; do
    M=$(basename "$f" .jsonl)
    [ "$M" = ar ] && continue
    echo "== $B: $M vs plain decoding"
    python scripts/diagnose_mismatch.py "$f" "$dir/ar.jsonl" --data-dir "$DATA" > "$OUT/${B}__${M}.txt" 2> "$OUT/${B}__${M}.err" \
      || echo "WARNING: $B $M failed (see $OUT/${B}__${M}.err)"
    echo "$B $M: $(tail -1 "$OUT/${B}__${M}.txt")" | tee -a "$OUT/summary.txt"
  done
done
echo "done: $OUT/summary.txt"
