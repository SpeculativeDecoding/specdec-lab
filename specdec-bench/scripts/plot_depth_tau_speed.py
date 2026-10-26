#!/usr/bin/env python
"""Speed vs acceptance by draft depth (width 10, budget 60), relative to EAGLE-3's depth 8.

  python scripts/plot_depth_tau_speed.py results/depth_grid_humaneval_gsm8k_mt_bench_h200 \
      --bench humaneval --out results/figures/depth_tau_vs_speed_humaneval_h200.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {"humaneval": "HumanEval", "gsm8k": "GSM8K", "mt_bench": "MT-Bench",
         "alpaca": "Alpaca", "sum": "Sum", "qa": "QA"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--bench", default="humaneval")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = {r["config"]: r for r in csv.DictReader(open(os.path.join(a.folder, "summary.csv")))
            if r["bench"] == a.bench and r["config"].startswith("eagle3_s") and r["config"].endswith("_k10_b60")}
    pts = sorted((int(r["steps"]) - 1, float(r["throughput_tok_s"]), float(r["tau"])) for r in rows.values())
    ref = next(p for p in pts if p[0] == 8)
    depth = [d for d, _, _ in pts]
    speed = [s / ref[1] for _, s, _ in pts]
    tau = [t / ref[2] for _, _, t in pts]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(depth, speed, "o-", color="tab:orange", label=f"Speed (tok/s); depth 8 = {ref[1]:.0f} tok/s")
    ax.plot(depth, tau, "s--", color="tab:blue", label=f"Tokens accepted per step (τ); depth 8 = {ref[2]:.2f}")
    ax.axvline(8, color="gray", linestyle=":", linewidth=1)
    ax.axhline(1.0, color="gray", linewidth=0.6)
    ax.annotate("EAGLE-3 default (depth 8)", xy=(8, 0.42), xytext=(8.4, 0.42), fontsize=8, color="gray")
    ax.set_xticks(depth)
    ax.set_xlabel("Draft depth (width 10, budget 60)")
    ax.set_ylabel("Relative to depth 8")
    ax.set_ylim(0.35, 1.18)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    name = NAMES.get(a.bench, a.bench)
    ax.set_title(f"{name}: acceptance keeps rising past depth 8, speed does not", fontsize=10)
    fig.text(0.01, 0.01, "EAGLE code base, Llama-3.1-8B + EAGLE-3, H200, batch 1, greedy; one run.",
             fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    for d, s, t in zip(depth, speed, tau):
        print(f"depth {d:2d}: speed {s:.3f}  tau {t:.3f}")
    print(f"wrote {a.out} and .pdf")


if __name__ == "__main__":
    main()
