#!/usr/bin/env python
"""Quick TALON vs EAGLE-3 comparison on a single prompt.

This is the simplest way to verify TALON is working. It runs:
  1. Plain autoregressive decoding (baseline)
  2. Static EAGLE-3 (K=10, D=8, N=60)
  3. TALON (N=60, mu=0.03, K=10)

And prints throughput, speedup, tau (acceptance rate), and delta (draft efficiency).

Usage:
    cd specdec-bench
    source .venv-eagle/bin/activate
    python scripts/quickstart_talon.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from specdec.adaptive import AdaptiveConfig
from specdec.tree_config import TreeConfig


PROMPT = """Write a Python function to calculate the nth Fibonacci number.
Include type hints, a docstring with examples, and handle edge cases.
Also write unit tests for the function."""


def main():
    # Import here so the script fails fast if no GPU
    import torch
    if not torch.cuda.is_available():
        print("ERROR: No GPU available. TALON benchmarks require CUDA.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    print()

    from specdec.eagle_runner import DEFAULT_BASE, DEFAULT_EAGLE3, load_eagle_model, EagleRunner

    # TALON paper settings: N=60, mu=0.03, K=10
    # EAGLE-3 paper baseline: K=10, D=8 (steps=9), N=60
    tree_cfg = TreeConfig(steps=9, topk=10, budget=60)
    adaptive_cfg = AdaptiveConfig(budget=60, mu=0.03, root_k=10)

    print("Loading model (this takes ~30s)...")

    # Load with TALON first (we'll compare both)
    model = load_eagle_model(None, DEFAULT_BASE, DEFAULT_EAGLE3,
                             use_eagle3=True, dtype="float16", adaptive=adaptive_cfg)
    runner = EagleRunner(model, max_new_tokens=512, max_length=4096, temperature=0.0)

    # Run TALON
    print("\nRunning TALON (adaptive tree)...")
    talon_result = runner._run_one(PROMPT, "spec")
    talon_output = talon_result["output"]
    talon_time = talon_result["seconds"]
    talon_tokens = talon_result["new_tokens"]
    talon_tau = talon_result["new_tokens"] / talon_result["verify_steps"]
    talon_delta = talon_result.get("draft_forwards", 0) / talon_result["verify_steps"]
    talon_depth = talon_result.get("mean_tree_depth", 0)

    # Run plain AR
    print("Running plain AR...")
    ar_result = runner._run_one(PROMPT, "ar")
    ar_time = ar_result["seconds"]
    ar_tokens = ar_result["new_tokens"]

    # Verify losslessness (outputs should match at T=0)
    ar_output = ar_result["output"]
    talon_matches_ar = (talon_output == ar_output)

    # Load static EAGLE-3 model
    print("Running EAGLE-3 (static tree)...")
    model_static = load_eagle_model(tree_cfg, DEFAULT_BASE, DEFAULT_EAGLE3,
                                    use_eagle3=True, dtype="float16", adaptive=None)
    runner_static = EagleRunner(model_static, max_new_tokens=512, max_length=4096, temperature=0.0)

    eagle_result = runner_static._run_one(PROMPT, "spec")
    eagle_time = eagle_result["seconds"]
    eagle_tokens = eagle_result["new_tokens"]
    eagle_tau = eagle_result["new_tokens"] / eagle_result["verify_steps"]
    eagle_delta = eagle_result.get("draft_forwards", 0) / eagle_result["verify_steps"]
    eagle_depth = eagle_result.get("mean_tree_depth", 0)
    eagle_output = eagle_result["output"]
    eagle_matches_ar = (eagle_output == ar_output)

    # Print results
    print("\n" + "=" * 70)
    print("TALON vs EAGLE-3 RESULTS")
    print("=" * 70)

    print(f"\n=== Output (TALON) ===")
    print(talon_output[:500] + "..." if len(talon_output) > 500 else talon_output)

    print(f"\nidentical to plain decoding:  EAGLE-3 {eagle_matches_ar} | TALON {talon_matches_ar}")

    print(f"\n{'Method':<20} {'Tokens':<10} {'tok/s':<10} {'tau':<8} {'delta':<8} {'depth':<8} {'speedup':<10}")
    print("-" * 80)

    ar_tps = ar_tokens / ar_time
    print(f"{'Plain AR':<20} {ar_tokens:<10} {ar_tps:<10.1f} {'-':<8} {'-':<8} {'-':<8} {'1.00x':<10}")

    eagle_tps = eagle_tokens / eagle_time
    print(f"{'EAGLE-3 (static)':<20} {eagle_tokens:<10} {eagle_tps:<10.1f} {eagle_tau:<8.2f} {eagle_delta:<8.2f} {eagle_depth:<8.1f} {eagle_tps/ar_tps:<9.2f}x")

    talon_tps = talon_tokens / talon_time
    print(f"{'TALON (adaptive)':<20} {talon_tokens:<10} {talon_tps:<10.1f} {talon_tau:<8.2f} {talon_delta:<8.2f} {talon_depth:<8.1f} {talon_tps/ar_tps:<9.2f}x")

    print()
    print("Key insights:")
    print(f"  - TALON draft efficiency (delta): {talon_delta:.2f} vs EAGLE-3: {eagle_delta:.2f}")
    print(f"    (Lower delta = fewer draft passes per verify, which is better)")
    print(f"  - TALON tree depth: {talon_depth:.1f} vs EAGLE-3: {eagle_depth:.1f}")
    print(f"    (TALON adapts depth based on confidence)")

    if talon_delta < eagle_delta:
        print(f"  - TALON uses {(1 - talon_delta/eagle_delta)*100:.1f}% fewer draft passes!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
