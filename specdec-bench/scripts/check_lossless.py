#!/usr/bin/env python
"""Losslessness check at T=0: compare speculative outputs with plain decoding.

  python scripts/check_lossless.py results/eagle/s6_k10_b60/mt_bench__s6_k10_b60__T0.jsonl \
                                   results/eagle/s6_k10_b60/mt_bench__ar__T0.jsonl

Exact token match is expected for most prompts but not guaranteed: fp16 kernels
reduce in a different order when verifying a tree than when decoding one token,
so near-ties in the logits can flip. Every mismatch should be inspected; a
mismatch rate well above a few percent points to a bug, not to numerics.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec.eagle_runner import read_jsonl  # noqa: E402


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spec_jsonl")
    p.add_argument("ar_jsonl")
    p.add_argument("--show", type=int, default=5, help="print the first N mismatches")
    args = p.parse_args()
    _, sd = read_jsonl(args.spec_jsonl)
    _, ar = read_jsonl(args.ar_jsonl)
    A = {r["qid_turn"]: r for r in sd}
    B = {r["qid_turn"]: r for r in ar}
    keys = sorted(set(A) & set(B), key=str)
    if not keys:
        sys.exit("no common prompts")
    use_ids = all("output_ids" in A[k] and "output_ids" in B[k] for k in keys)
    same, mism = 0, []
    for k in keys:
        a = A[k]["output_ids"] if use_ids else A[k]["text"]
        b = B[k]["output_ids"] if use_ids else B[k]["text"]
        if a == b:
            same += 1
        else:
            mism.append((k, first_divergence(a, b), len(a), len(b)))
    unit = "tokens" if use_ids else "chars"
    print(f"compared {len(keys)} turns by {'token ids' if use_ids else 'text'}: "
          f"{same} identical ({100 * same / len(keys):.1f}%), {len(mism)} differ")
    # Later turns of a multi-turn chat inherit earlier divergence; report turn-0 separately.
    t0 = [k for k in keys if k.endswith(":0")]
    if len(t0) != len(keys):
        s0 = sum(1 for k in t0 if (A[k].get("output_ids", A[k]["text"]) == B[k].get("output_ids", B[k]["text"])))
        print(f"first turns only: {s0}/{len(t0)} identical")
    for k, pos, la, lb in mism[: args.show]:
        print(f"  {k}: first difference at {unit} {pos} (lengths {la} vs {lb})")
    only = (set(A) ^ set(B))
    if only:
        print(f"note: {len(only)} prompts present in only one file")


if __name__ == "__main__":
    main()
