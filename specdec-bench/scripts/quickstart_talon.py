#!/usr/bin/env python
"""One prompt, three ways, one model load: plain decoding vs EAGLE-3 vs TALON.

  python scripts/quickstart_talon.py
  python scripts/quickstart_talon.py --prompt "Write quicksort in Python." --mu 0.03 --budget 60

EAGLE-3 uses the TALON paper's baseline tree (K=10, D=8, N=60 -> steps 9, topk 10).
TALON uses the paper defaults (N=60, mu=0.03, K=10) and the same draft weights. Both
verify N nodes, so the comparison is at a matched budget.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec.data import encode_messages, initial_messages  # noqa: E402
from specdec.eagle_runner import (DEFAULT_BASE, DEFAULT_EAGLE3, EagleRunner,  # noqa: E402
                                  instrument_drafting, load_eagle_model)
from specdec.talon import TalonConfig, enable_talon  # noqa: E402
from specdec.tree_config import TreeConfig  # noqa: E402


def show(name, r, ar=None):
    tps = r["output_tokens"] / r["wall_time"]
    line = f"{name:8s} {r['output_tokens']:4d} tok  {tps:7.1f} tok/s"
    if "tau" in r and name != "plain":
        line += f"  tau={r['tau']:.2f}"
    if "delta" in r:
        line += f"  delta={r['delta']:.2f}  tree depth={r['mean_tree_depth']:.1f}"
    if ar is not None:
        line += f"  speedup={tps / (ar['output_tokens'] / ar['wall_time']):.2f}x"
    print(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Write a Python function that returns the n-th Fibonacci number, with a docstring and tests.")
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--ea-model", default=DEFAULT_EAGLE3)
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--mu", type=float, default=0.03)
    p.add_argument("--root-k", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    baseline = TreeConfig(9, 10, args.budget, "talon_eagle3_baseline").validate()
    model = load_eagle_model(baseline, args.base_model, args.ea_model)   # instrumented static tree
    runner = EagleRunner(model, max_new_tokens=args.max_new_tokens)
    ids = encode_messages(runner.tok, initial_messages() + [{"role": "user", "content": args.prompt}])

    for _ in range(2):                                   # warm-up, allocates the KV cache once
        runner.generate(ids, "spec")
        runner.generate(ids, "ar")
    ar = runner.generate(ids, "ar")
    eagle = runner.generate(ids, "spec")

    enable_talon(model, TalonConfig(args.budget, args.mu, args.root_k))  # same weights, same N
    instrument_drafting(model, static_depth=None)
    runner.generate(ids, "spec")                         # warm-up the TALON path
    talon = runner.generate(ids, "spec")

    print("\n=== TALON output ===\n" + talon["text"] + "\n")
    print("identical to plain decoding:  EAGLE-3", eagle["output_ids"] == ar["output_ids"],
          "| TALON", talon["output_ids"] == ar["output_ids"])
    show("plain", ar)
    show("EAGLE-3", eagle, ar)
    show("TALON", talon, ar)


if __name__ == "__main__":
    main()
