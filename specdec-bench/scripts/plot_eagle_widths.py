#!/usr/bin/env python
"""Realised EAGLE-3 tree shape: mean verified nodes at each depth, one line per workload.

Reads the per-depth CSV written by `plot_trees.py widths`.

  python scripts/plot_eagle_widths.py results/figures/widths_eagle_vs_talon_h200.csv \
      --benches humaneval,qa --out results/figures/eagle3_widths_humaneval_qa_h200.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {"humaneval": "Code (HumanEval)", "gsm8k": "Math (GSM8K)", "mt_bench": "MT-Bench",
         "alpaca": "Alpaca", "sum": "Summarisation", "qa": "QA (Natural Questions)"}
STYLES = ["o-", "s--", "^-.", "D:"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--benches", default="humaneval,qa")
    p.add_argument("--config", default="eagle3_s9_k10_b60")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = [r for r in csv.DictReader(open(a.csv)) if r["config"] == a.config]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, b in enumerate(a.benches.split(",")):
        pts = sorted((int(r["depth"]), float(r["mean_width"])) for r in rows if r["bench"] == b)
        d, w = zip(*pts)
        ax.plot(d, w, STYLES[i], color=COLORS[i], label=f"{NAMES.get(b, b)}  (total {sum(w):.0f})")
        print(b, " ".join(f"{x:.2f}" for x in w), f"sum {sum(w):.1f}")
    ax.set_xticks(d)
    ax.set_xlabel("Depth below the root")
    ax.set_ylabel("Mean nodes verified at this depth")
    ax.set_ylim(0, 11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("EAGLE-3 default tree (8 × 10, budget 60): realised shape by workload", fontsize=10)
    fig.text(0.01, 0.01, "EAGLE code base, Llama-3.1-8B + EAGLE-3, H200, batch 1, greedy; "
             "averaged over all decoding steps; one run.", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print(f"wrote {a.out} and .pdf")


if __name__ == "__main__":
    main()
