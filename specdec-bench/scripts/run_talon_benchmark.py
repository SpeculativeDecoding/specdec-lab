#!/usr/bin/env python
"""TALON vs EAGLE-3 benchmark script.

This script implements the benchmark logic from the TALON paper (arXiv 2601.07353):
- Compares TALON (adaptive draft trees) vs static EAGLE-3
- Measures acceptance rate, speedup, and draft efficiency
- Supports batch size 1 (Path B: official EAGLE repo)

Usage:
    # Quick smoke test (10 questions, MT-Bench only)
    python scripts/run_talon_benchmark.py --data-dir $EAGLE_DATA --bench mt_bench --limit 10 \
        --out-dir results/talon_smoke

    # Full benchmark on all datasets
    python scripts/run_talon_benchmark.py --data-dir $EAGLE_DATA --bench all \
        --out-dir results/talon_full

    # Phase 2: Check how exact TALON trees are (shape analysis)
    python scripts/run_talon_benchmark.py --data-dir $EAGLE_DATA --bench all --limit 10 \
        --talon --shape-analysis --out-dir results/talon_shape
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from specdec.data import BENCHES, load_questions, parse_benches, select
from specdec.eagle_runner import (DEFAULT_BASE, DEFAULT_EAGLE3, EagleRunner,
                                  load_eagle_model, read_jsonl)
from specdec.metrics import pooled_tau, throughput
from specdec.adaptive import AdaptiveConfig
from specdec.tree_config import TreeConfig


def git_sha(path: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def analyze_tree_shapes(jsonl_path: str) -> Dict:
    """Analyze tree shapes from TALON run to check SGLang compatibility."""
    _, rows = read_jsonl(jsonl_path)

    tree_calls = 0
    depth_hist = defaultdict(int)
    width_hist = defaultdict(int)
    max_width_per_tree = []
    wide_trees = 0  # trees with any layer wider than 10
    deep_trees = 0  # trees with depth > 9

    for row in rows:
        if "tree_depth_hist" not in row:
            continue
        for depth, count in row.get("tree_depth_hist", {}).items():
            depth_hist[int(depth)] += count
            tree_calls += count

        widths = row.get("tree_widths", [])
        if widths:
            max_w = max(widths) if widths else 0
            max_width_per_tree.append(max_w)
            if max_w > 10:
                wide_trees += 1

        mean_depth = row.get("mean_tree_depth", 0)
        if mean_depth > 9:
            deep_trees += 1

    # Compute percentages
    depth_le_9 = sum(v for k, v in depth_hist.items() if k <= 9)
    pct_depth_le_9 = (depth_le_9 / tree_calls * 100) if tree_calls > 0 else 0
    pct_width_le_10 = ((tree_calls - wide_trees) / tree_calls * 100) if tree_calls > 0 else 0

    return {
        "tree_calls": tree_calls,
        "depth_hist": dict(depth_hist),
        "pct_depth_le_9": pct_depth_le_9,
        "pct_width_le_10": pct_width_le_10,
        "wide_trees": wide_trees,
        "deep_trees": deep_trees,
        "max_layer_width": max(max_width_per_tree) if max_width_per_tree else 0,
    }


def run_single_config(
    runner: EagleRunner,
    questions: List[Dict],
    mode: str,
    out_path: str,
    meta: Dict,
    warmup: int = 3,
    max_turns: int = 0,
    skip_existing: bool = False,
    no_resume: bool = False,
) -> Optional[List[Dict]]:
    """Run a single configuration and return results."""
    if skip_existing and os.path.exists(out_path):
        print(f"[{meta['bench']}:{mode}] exists, skipping -> {out_path}", flush=True)
        return None

    rows = runner.run(
        questions, mode, out_path, meta=meta, warmup=warmup,
        max_turns=max_turns, resume=not no_resume
    )

    msg = f"[{meta['bench']}:{mode}] {len(rows)} turns, {throughput(rows):.1f} tok/s"
    if mode == "spec":
        msg += f", tau={pooled_tau(rows):.3f}"
        fw = sum(r.get("draft_forwards", 0) for r in rows)
        vs = sum(r["verify_steps"] for r in rows)
        if fw:
            msg += f", delta={fw / vs:.2f}"
    print(msg, "->", out_path, flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Data and benchmark selection
    p.add_argument("--data-dir", required=True, help="EAGLE/eagle/data directory")
    p.add_argument("--bench", default="mt_bench",
                   help=f"Benchmark: {list(BENCHES)}, comma-separated, or 'all'")
    p.add_argument("--subset", default="all", choices=["all", "calib", "test"])
    p.add_argument("--limit", type=int, default=0, help="First N questions per bench (0=all)")
    p.add_argument("--max-turns", type=int, default=0, help="Max turns per question (0=all)")

    # Model configuration
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--ea-model", default=DEFAULT_EAGLE3)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])

    # TALON configuration (paper defaults: N=60, mu=0.03, K=10)
    p.add_argument("--talon", action="store_true", help="Run TALON (adaptive tree)")
    p.add_argument("--budget", type=int, default=60, help="TALON budget N (verified nodes incl. root)")
    p.add_argument("--mu", type=float, default=0.03, help="TALON confidence threshold")
    p.add_argument("--root-k", type=int, default=10, help="TALON top-K width of root layer")

    # Static EAGLE-3 configuration (paper baseline: K=10, D=8)
    p.add_argument("--static", action="store_true", help="Run static EAGLE-3")
    p.add_argument("--steps", type=int, default=9, help="EAGLE tree depth in layers")
    p.add_argument("--topk", type=int, default=10, help="EAGLE children per parent")

    # AR baseline
    p.add_argument("--ar", action="store_true", help="Run plain AR decoding")
    p.add_argument("--with-ar", action="store_true", help="Also run AR after spec")

    # Run options
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-resume", action="store_true")

    # Output
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--eagle-repo", default=os.environ.get("EAGLE_REPO", ""))

    # Analysis
    p.add_argument("--shape-analysis", action="store_true",
                   help="Analyze TALON tree shapes for SGLang compatibility")

    args = p.parse_args()

    # Parse benchmarks
    try:
        benches = parse_benches(args.bench)
    except ValueError as exc:
        p.error(str(exc))

    # Determine what to run
    if not any([args.talon, args.static, args.ar]):
        # Default: run both TALON and static for comparison
        args.talon = True
        args.static = True
        args.with_ar = True

    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    import transformers

    # Build run configurations
    configs = []

    if args.talon:
        adaptive = AdaptiveConfig(args.budget, args.mu, args.root_k).validate()
        configs.append({
            "name": adaptive.name,
            "mode": "spec",
            "tree": None,
            "adaptive": adaptive,
            "meta_key": "adaptive",
        })

    if args.static:
        tree = TreeConfig(args.steps, args.topk, args.budget).validate()
        configs.append({
            "name": tree.name,
            "mode": "spec",
            "tree": tree,
            "adaptive": None,
            "meta_key": "tree",
        })

    if args.ar or args.with_ar:
        configs.append({
            "name": "ar",
            "mode": "ar",
            "tree": None,
            "adaptive": None,
            "meta_key": None,
        })

    # Load model once for all runs
    first_tree = next((c["tree"] for c in configs if c["tree"]), None)
    first_adaptive = next((c["adaptive"] for c in configs if c["adaptive"]), None)

    model = load_eagle_model(
        first_tree, args.base_model, args.ea_model,
        use_eagle3=True, dtype=args.dtype, adaptive=first_adaptive
    )
    runner = EagleRunner(
        model, max_new_tokens=args.max_new_tokens,
        max_length=args.max_length, temperature=args.temperature
    )

    results = defaultdict(dict)

    for bench in benches:
        qs = select(load_questions(args.data_dir, bench), args.subset, args.limit)
        print(f"\n=== {bench}: {len(qs)} questions ===", flush=True)

        for cfg in configs:
            # Build metadata
            meta = {
                "path": "eagle_repo",
                "bench": bench,
                "mode": cfg["mode"],
                "subset": args.subset,
                "tree": cfg["tree"].to_dict() if cfg["tree"] else None,
                "adaptive": cfg["adaptive"].to_dict() if cfg["adaptive"] else None,
                "base_model": args.base_model,
                "ea_model": args.ea_model,
                "use_eagle3": True,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "max_length": args.max_length,
                "dtype": args.dtype,
                "batch_size": 1,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "python": platform.python_version(),
                "eagle_commit": git_sha(args.eagle_repo) if args.eagle_repo else None,
            }

            out_path = os.path.join(
                args.out_dir,
                f"{bench}__{cfg['name']}__T{args.temperature:g}.jsonl"
            )

            # Reload model with correct config if needed
            if cfg["adaptive"] and not hasattr(model.ea_layer, "adaptive"):
                model = load_eagle_model(
                    None, args.base_model, args.ea_model,
                    use_eagle3=True, dtype=args.dtype, adaptive=cfg["adaptive"]
                )
                runner = EagleRunner(
                    model, max_new_tokens=args.max_new_tokens,
                    max_length=args.max_length, temperature=args.temperature
                )

            rows = run_single_config(
                runner, qs, cfg["mode"], out_path, meta,
                warmup=args.warmup, max_turns=args.max_turns,
                skip_existing=args.skip_existing, no_resume=args.no_resume
            )

            if rows:
                results[bench][cfg["name"]] = {
                    "path": out_path,
                    "throughput": throughput(rows),
                    "tau": pooled_tau(rows) if cfg["mode"] == "spec" else None,
                    "n_turns": len(rows),
                }

                # Shape analysis for TALON
                if args.shape_analysis and cfg["adaptive"]:
                    shape = analyze_tree_shapes(out_path)
                    results[bench][cfg["name"]]["shape"] = shape
                    print(f"  Shape analysis: {shape['tree_calls']} trees, "
                          f"{shape['pct_depth_le_9']:.1f}% depth<=9, "
                          f"{shape['pct_width_le_10']:.1f}% width<=10, "
                          f"max_width={shape['max_layer_width']}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: TALON vs EAGLE-3 Benchmark")
    print("=" * 60)

    for bench in benches:
        if bench not in results:
            continue
        print(f"\n{bench}:")
        ar_tps = results[bench].get("ar", {}).get("throughput", 0)
        for name, data in results[bench].items():
            tps = data["throughput"]
            tau = data.get("tau")
            speedup = tps / ar_tps if ar_tps > 0 else 0
            line = f"  {name:30s} {tps:7.1f} tok/s"
            if tau:
                line += f"  tau={tau:.2f}"
            if ar_tps > 0 and name != "ar":
                line += f"  speedup={speedup:.2f}x"
            print(line)

    # Write summary JSON
    summary_path = os.path.join(args.out_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(dict(results), f, indent=2, default=str)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
