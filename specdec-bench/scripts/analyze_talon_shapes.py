#!/usr/bin/env python
"""How many TALON trees fit SGLang's fixed EAGLE tree (TALON-S's envelope)?

SGLang drafts `steps` layers and keeps the `topk` best nodes of each layer as the frontier, each
with up to `topk` children. A TALON tree fits if its depth <= steps, it expands at most `topk`
nodes in any layer and gives no node more than `topk` children. Fitting is necessary for TALON-S
to reproduce the tree, not sufficient (SGLang also picks which nodes form the frontier).

  python scripts/analyze_talon_shapes.py results/talon_shapes_humaneval_gsm8k_mt_bench_h200 --out shapes.csv
"""
import argparse
import csv
import glob
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+")
    p.add_argument("--steps", type=int, default=9)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--out")
    a = p.parse_args()
    rows = []
    for folder in a.folders:
        for f in sorted(glob.glob(os.path.join(folder, "**", "*.jsonl"), recursive=True)):
            shapes, bench = [], None
            for line in open(f):
                r = json.loads(line)
                if "tree_shapes" in r:
                    bench = r.get("bench") or os.path.basename(f).split("__")[0]
                    shapes += r["tree_shapes"]
            if not shapes:
                continue
            n = len(shapes)
            depth_ok = [s[0] <= a.steps for s in shapes]
            expand_ok = [s[2] <= a.topk for s in shapes]
            child_ok = [s[3] <= a.topk for s in shapes]
            fits = [x and y and z for x, y, z in zip(depth_ok, expand_ok, child_ok)]
            rows.append({
                "bench": bench, "trees": n,
                "mean_depth": round(sum(s[0] for s in shapes) / n, 2),
                "max_depth": max(s[0] for s in shapes),
                "mean_widest_layer": round(sum(s[1] for s in shapes) / n, 2),
                f"depth_le_{a.steps}": round(sum(depth_ok) / n, 4),
                f"expanded_le_{a.topk}": round(sum(expand_ok) / n, 4),
                f"children_le_{a.topk}": round(sum(child_ok) / n, 4),
                f"widest_le_{a.topk}": round(sum(s[1] <= a.topk for s in shapes) / n, 4),
                "fits_sglang": round(sum(fits) / n, 4),
            })
    for r in rows:
        print(r)
    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
