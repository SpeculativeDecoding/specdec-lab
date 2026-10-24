#!/usr/bin/env python
"""EAGLE-3 vs TALON vs "Ours" (a fixed tree chosen per workload offline), relative to EAGLE-3.

  python scripts/compare_vs_eagle3.py results/depth_grid_* results/dxw80_sweep_* results/width_grid_* \
      --out results/summary_vs_eagle3.csv

Path B (EAGLE repo, batch 1) folders only. Each folder holds plain decoding (`ar`) and, except the
width grid, EAGLE-3's 8 x 10 tree (`s9_k10_b60`) from the same model load; TALON comes from the
depth-grid folders.

"Ours": for every workload, the fixed tree with the highest speedup over plain decoding on the
calibration questions (frozen 30% split, specdec.data.split) among every fixed tree run on that
workload, EAGLE-3's own 8 x 10 included. It is then scored only on the other 70% (test).

All ratios are throughputs on the test questions. A tree is compared with the EAGLE-3 run of its
own folder; for the width grid (no 8 x 10 of its own) the two speedups over each folder's own
plain decoding are divided. The 95% CI is a paired bootstrap over test questions, resampling
whole questions (both MT-Bench turns together).
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec.data import _bucket  # noqa: E402

EAGLE = "s9_k10_b60"
TALON = "adaptive_n60_mu0.03_k10"
CALIB_FRAC = 0.3


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "_meta" not in d:
                rows.append(d)
    return rows


def tree_label(cfg):
    """s13_k4_b60 -> '12 x 4' (EAGLE depth = steps - 1)."""
    if cfg == TALON:
        return "TALON"
    s, k, _ = cfg.split("_")
    return f"{int(s[1:]) - 1} x {int(k[1:])}"


def collect(folders):
    """runs[(folder, bench, cfg)] = {question_id: (tokens, seconds)}; cfg 'ar' = plain decoding."""
    runs = {}
    for folder in folders:
        for path in glob.glob(os.path.join(folder, "*", "*__T0.jsonl")):
            bench, cfg = os.path.basename(path).split("__")[:2]
            per_q = {}
            for r in load(path):
                t, w = per_q.get(r["question_id"], (0.0, 0.0))
                per_q[r["question_id"]] = (t + r["output_tokens"], w + r["wall_time"])
            runs[(os.path.basename(folder.rstrip("/")), bench, cfg)] = per_q
    return runs


def arrays(run, qids):
    return np.array([run[q][0] for q in qids]), np.array([run[q][1] for q in qids])


def speedup(runs, folder, bench, cfg, qids):
    t, w = arrays(runs[(folder, bench, cfg)], qids)
    at, aw = arrays(runs[(folder, bench, "ar")], qids)
    return (t.sum() / w.sum()) / (at.sum() / aw.sum())


def ratio_with_ci(runs, num, den, qids, n_boot, rng):
    """num, den = (folder, bench, cfg); ratio of speedups over each folder's own plain decoding."""
    parts = []
    for folder, bench, cfg in (num, den):
        parts += [arrays(runs[(folder, bench, cfg)], qids), arrays(runs[(folder, bench, "ar")], qids)]
    (nt, nw), (nat, naw), (dt, dw), (dat, daw) = parts

    def r(ix):
        s = lambda t, w: t[ix].sum(-1) / w[ix].sum(-1)  # noqa: E731
        return (s(nt, nw) / s(nat, naw)) / (s(dt, dw) / s(dat, daw))

    point = float(r(np.arange(len(qids))))
    boot = r(rng.integers(0, len(qids), size=(n_boot, len(qids))))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return point, float(lo), float(hi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()
    runs = collect(args.folders)
    rng = np.random.default_rng(0)

    benches = sorted({b for _, b, _ in runs})
    out = []
    for bench in benches:
        ref_folders = [f for f, b, c in runs if b == bench and c == EAGLE]
        talon_folders = [f for f, b, c in runs if b == bench and c == TALON]
        if not ref_folders:
            continue
        ref = ref_folders[0] if not talon_folders else talon_folders[0]  # depth grid when it has the bench
        qids = sorted(runs[(ref, bench, EAGLE)], key=str)
        calib = [q for q in qids if _bucket(bench, q) < CALIB_FRAC]
        test = [q for q in qids if _bucket(bench, q) >= CALIB_FRAC]

        # Ours: best fixed tree on the calibration questions, over every folder that ran this bench
        cands = [(f, c) for f, b, c in runs if b == bench and c not in ("ar", TALON)]
        best_f, best_c = max(cands, key=lambda fc: speedup(runs, fc[0], bench, fc[1], calib))
        own_ref = best_f if (best_f, bench, EAGLE) in runs else ref

        rows = [("EAGLE-3 (8 x 10)", (ref, bench, EAGLE))]
        if talon_folders:
            rows.append(("TALON", (talon_folders[0], bench, TALON)))
        rows.append((f"Ours ({tree_label(best_c)})", (best_f, bench, best_c)))
        for name, key in rows:
            den = (own_ref, bench, EAGLE) if name.startswith("Ours") else (key[0] if (key[0], bench, EAGLE) in runs else ref, bench, EAGLE)
            vs, lo, hi = ratio_with_ci(runs, key, den, test, args.n_boot, rng)
            out.append({"bench": bench, "method": name, "folder": key[0], "config": key[2],
                        "n_calib": len(calib), "n_test": len(test),
                        "speedup_vs_plain_test": round(speedup(runs, *key, test), 3),
                        "vs_eagle3_test": round(vs, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows to {args.out}")
    for r in out:
        print(f"{r['bench']:10s} {r['method']:22s} plain x{r['speedup_vs_plain_test']:.2f}  "
              f"vs EAGLE-3 {r['vs_eagle3_test']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  ({r['folder']})")


if __name__ == "__main__":
    main()
