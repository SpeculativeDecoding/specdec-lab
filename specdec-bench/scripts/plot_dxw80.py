#!/usr/bin/env python
"""Speed across the depth x width = 80 sweep, relative to each workload's 8 x 10 tree.

  python scripts/plot_dxw80.py results/dxw80_sweep_{humaneval,gsm8k,sum,qa}_h200 \
      --out results/figures/dxw80_speed_h200.png
"""
import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {"humaneval": "HumanEval", "gsm8k": "GSM8K", "mt_bench": "MT-Bench",
         "alpaca": "Alpaca", "sum": "Sum", "qa": "QA"}
STYLES = ["o-", "s--", "^-.", "D:"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, folder in enumerate(a.folders):
        rows = list(csv.DictReader(open(os.path.join(folder, "summary.csv"))))
        bench = rows[0]["bench"]
        ref = next(float(r["throughput_tok_s"]) for r in rows if r["config"] == "eagle3_s9_k10_b60")
        pts = []
        for r in rows:
            m = re.fullmatch(r"eagle3_s(\d+)_k(\d+)_b60", r["config"])
            if m:
                pts.append((int(m[1]) - 1, int(m[2]), float(r["throughput_tok_s"]) / ref))
        pts.sort()
        labels = [f"{d}×{w}" for d, w, _ in pts]
        ax.plot(range(len(pts)), [s for _, _, s in pts], STYLES[i], color=COLORS[i],
                markersize=5, label=NAMES.get(bench, bench))
        print(bench, " ".join(f"{l}:{s:.3f}" for l, (_, _, s) in zip(labels, pts)))
    ax.set_xticks(range(len(labels)), labels)
    ax.axhline(1.0, color="gray", linewidth=0.6)
    ax.set_xlabel("Draft depth × width (depth × width = 80; deeper and narrower to the right)")
    ax.set_ylabel("Speed relative to 8 × 10")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    ax.set_title("Speed peaks at 8 × 10 for code and math, at 5 × 16 for Sum and QA", fontsize=10)
    fig.text(0.01, 0.01, "EAGLE code base, Llama-3.1-8B + EAGLE-3, budget 60, H200, batch 1, greedy; "
             "one run (repeats differ by 2–3%).", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print(f"wrote {a.out} and .pdf")


if __name__ == "__main__":
    main()
