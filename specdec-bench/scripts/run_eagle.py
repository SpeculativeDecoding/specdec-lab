#!/usr/bin/env python
"""Path B: EAGLE-3 (or plain AR) on Llama-3.1-8B-Instruct with the official EAGLE repo.

Examples
  # EAGLE repo's own default tree (total_token=60, depth=5, top_k=10) on MT-Bench
  python scripts/run_eagle.py --data-dir ~/EAGLE/eagle/data --bench mt_bench \
      --steps 6 --topk 10 --budget 60 --out results/eagle/mt_bench_s6_k10_b60.jsonl

  # Plain autoregressive baseline on the same prompts (same model object, naivegenerate)
  python scripts/run_eagle.py --data-dir ~/EAGLE/eagle/data --bench mt_bench --ar \
      --out results/eagle/mt_bench_ar.jsonl

  # Several benches and the AR baseline in ONE model load (avoids reloading 16 GB)
  python scripts/run_eagle.py --data-dir ~/EAGLE/eagle/data --bench all --with-ar \
      --steps 6 --topk 10 --budget 60 --out-dir results/eagle/s6_k10_b60

  # TALON (paper defaults: N=60, mu=0.03, K=10) on the same EAGLE-3 drafter
  python scripts/run_eagle.py --data-dir ~/EAGLE/eagle/data --bench all --talon \
      --budget 60 --mu 0.03 --root-k 10 --out-dir results/eagle/talon_n60_mu0.03_k10
"""
import argparse
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from specdec.data import BENCHES, load_questions, select  # noqa: E402
from specdec.eagle_runner import (DEFAULT_BASE, DEFAULT_EAGLE3, EagleRunner,  # noqa: E402
                                  load_eagle_model)
from specdec.metrics import pooled_tau, throughput  # noqa: E402
from specdec.talon import TalonConfig  # noqa: E402
from specdec.tree_config import TreeConfig  # noqa: E402


def git_sha(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="EAGLE/eagle/data")
    p.add_argument("--bench", default="mt_bench", help=f"one of {BENCHES} or 'all'")
    p.add_argument("--subset", default="all", choices=["all", "calib", "test"])
    p.add_argument("--limit", type=int, default=0, help="first N questions per bench (0 = all)")
    p.add_argument("--max-turns", type=int, default=0, help="0 = all turns (MT-Bench has 2)")
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--ea-model", default=DEFAULT_EAGLE3)
    p.add_argument("--eagle1", action="store_true", help="draft is EAGLE-1 weights (use_eagle3=False)")
    p.add_argument("--steps", type=int, default=6, help="tree depth in layers (SGLang num_steps)")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--budget", type=int, default=60, help="verified nodes incl. root (SGLang num_draft_tokens; TALON N)")
    p.add_argument("--talon", action="store_true", help="build draft trees with TALON (EAGLE-3 only)")
    p.add_argument("--mu", type=float, default=0.03, help="TALON confidence threshold")
    p.add_argument("--root-k", type=int, default=10, help="TALON top-K width of the root layer")
    p.add_argument("--talon-max-depth", type=int, default=0, help="optional TALON depth cap (0 = none)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-length", type=int, default=4096, help="KV cache size; keep fixed across runs")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--ar", action="store_true", help="run only plain AR decoding")
    p.add_argument("--with-ar", action="store_true", help="also run AR after spec, same model load")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--skip-existing", action="store_true",
                   help="skip outputs that already exist (resume after an interruption)")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore a <out>.partial checkpoint and rerun that file from the start")
    p.add_argument("--out", help="output jsonl (single bench)")
    p.add_argument("--out-dir", help="output directory (bench 'all' or --with-ar)")
    p.add_argument("--eagle-repo", default=os.environ.get("EAGLE_REPO", ""), help="for recording the commit")
    args = p.parse_args()

    benches = list(BENCHES) if args.bench == "all" else [args.bench]
    if (len(benches) > 1 or args.with_ar) and not args.out_dir:
        p.error("--out-dir is required with --bench all or --with-ar")
    if len(benches) == 1 and not args.with_ar and not (args.out or args.out_dir):
        p.error("give --out or --out-dir")

    if args.talon:
        if args.eagle1:
            p.error("--talon needs an EAGLE-3 drafter")
        talon = TalonConfig(args.budget, args.mu, args.root_k, args.talon_max_depth).validate()
        tree = None
        spec_name, spec_meta = talon.name, {"talon": talon.to_dict()}
    else:
        talon = None
        tree = TreeConfig(args.steps, args.topk, args.budget).validate()
        spec_name, spec_meta = tree.name, {"tree": tree.to_dict()}
    model = load_eagle_model(tree, args.base_model, args.ea_model,
                             use_eagle3=not args.eagle1, dtype=args.dtype, talon=talon)
    runner = EagleRunner(model, max_new_tokens=args.max_new_tokens,
                         max_length=args.max_length, temperature=args.temperature)

    import torch
    import transformers
    modes = ["ar"] if args.ar else (["spec", "ar"] if args.with_ar else ["spec"])
    for bench in benches:
        qs = select(load_questions(args.data_dir, bench), args.subset, args.limit)
        for mode in modes:
            meta = {"path": "eagle_repo", "bench": bench, "mode": mode, "subset": args.subset,
                    "tree": spec_meta.get("tree") if mode == "spec" else None,
                    "talon": spec_meta.get("talon") if mode == "spec" else None,
                    "base_model": args.base_model, "ea_model": args.ea_model,
                    "use_eagle3": not args.eagle1, "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens, "max_length": args.max_length,
                    "dtype": args.dtype, "batch_size": 1,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "python": platform.python_version(),
                    "eagle_commit": git_sha(args.eagle_repo) if args.eagle_repo else None}
            if args.out_dir:
                tag = spec_name if mode == "spec" else "ar"
                out = os.path.join(args.out_dir, f"{bench}__{tag}__T{args.temperature:g}.jsonl")
            else:
                out = args.out
            if args.skip_existing and os.path.exists(out):
                print(f"[{bench}:{mode}] exists, skipping -> {out}", flush=True)
                continue
            rows = runner.run(qs, mode, out, meta=meta, warmup=args.warmup, max_turns=args.max_turns,
                            resume=not args.no_resume)
            msg = f"[{bench}:{mode}] {len(rows)} turns, {throughput(rows):.1f} tok/s"
            if mode == "spec":
                msg += f", tau={pooled_tau(rows):.3f}"
                fw = sum(r.get("draft_forwards", 0) for r in rows)
                vs = sum(r["verify_steps"] for r in rows)
                if fw:
                    msg += f", delta={fw / vs:.2f}"
            risky = sum(r["length_risk"] for r in rows)
            if risky:
                msg += f"  WARNING: {risky} turns may hit max_length={args.max_length}"
            print(msg, "->", out, flush=True)


if __name__ == "__main__":
    main()
