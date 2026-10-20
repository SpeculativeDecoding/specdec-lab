#!/usr/bin/env python
"""Smallest end-to-end check: EAGLE-3 vs plain decoding on Llama-3.1-8B-Instruct (one prompt).

  python scripts/quickstart_eagle3.py
  python scripts/quickstart_eagle3.py --prompt "Write a Python function for quicksort." --steps 8 --topk 10 --budget 60

Prints both outputs, whether they are identical (expected at T=0, up to fp16 ties),
tau (tokens per target forward pass) and the measured speedup.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec.data import encode_messages, initial_messages  # noqa: E402
from specdec.eagle_runner import DEFAULT_BASE, DEFAULT_EAGLE3, EagleRunner, load_eagle_model  # noqa: E402
from specdec.tree_config import TreeConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Explain how speculative decoding speeds up LLM inference, step by step.")
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--ea-model", default=DEFAULT_EAGLE3)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    tree = TreeConfig(args.steps, args.topk, args.budget).validate()
    print(f"tree: steps={tree.steps} topk={tree.topk} budget={tree.num_draft_tokens} "
          f"(EAGLE repo: total_token={tree.eagle_total_token}, depth={tree.eagle_depth}, top_k={tree.eagle_top_k}); "
          f"candidate pool={tree.pool}")
    model = load_eagle_model(tree, args.base_model, args.ea_model, use_eagle3=True)
    runner = EagleRunner(model, max_new_tokens=args.max_new_tokens)
    msgs = initial_messages() + [{"role": "user", "content": args.prompt}]
    ids = encode_messages(runner.tok, msgs)

    for _ in range(2):  # warm-up (allocates the KV cache once, compiles kernels)
        runner.generate(ids, "spec")
        runner.generate(ids, "ar")
    sd = runner.generate(ids, "spec")
    ar = runner.generate(ids, "ar")

    print("\n=== EAGLE-3 output ===\n" + sd["text"])
    print("\n=== identical to plain decoding:", sd["output_ids"] == ar["output_ids"])
    print(f"EAGLE-3: {sd['output_tokens']} tokens in {sd['wall_time']:.2f}s "
          f"({sd['output_tokens'] / sd['wall_time']:.1f} tok/s), tau={sd['tau']:.2f} "
          f"over {sd['verify_steps']} verify passes")
    print(f"Plain  : {ar['output_tokens']} tokens in {ar['wall_time']:.2f}s "
          f"({ar['output_tokens'] / ar['wall_time']:.1f} tok/s)")
    print(f"Speedup: {(sd['output_tokens'] / sd['wall_time']) / (ar['output_tokens'] / ar['wall_time']):.2f}x")


if __name__ == "__main__":
    main()
