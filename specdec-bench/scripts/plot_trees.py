#!/usr/bin/env python
"""Pictures of the draft trees that were actually verified (Path B runs).

  # mean width at each depth, one panel per workload, one line per tree configuration
  python scripts/plot_trees.py widths results/talon_depthcap_v2_*_h200 --out results/figures/widths.png \
      --configs eagle3_s9_k10_b60 eagle3_s6_k16_b60 adaptive_n60_mu0.03_k10 adaptive_n60_mu0.03_k10_d6_fill

  # example trees (runs made with --save-trees N): node-link drawings, one per configuration
  python scripts/plot_trees.py trees results/<run folder> --bench qa --out results/figures/qa_trees.png

`widths` reads `mean_layer_nodes` (averaged over turns, weighted by tree_calls) and also writes
the numbers as <out>.csv. `trees` reads `sample_trees` (parents and depths of the first trees of
each turn) and draws the tree of the given turn and step for each configuration.
"""
import argparse
import csv
import glob
import json
import os

# Validated default categorical palette (dataviz skill, references/palette.md), in fixed order.
# Colour follows the configuration: the same tree keeps its colour in every panel.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

LABELS = {
    "eagle3_s9_k10_b60": "EAGLE-3 8×10",
    "eagle3_s6_k16_b60": "EAGLE-3 5×16",
    "adaptive_n60_mu0.03_k10": "TALON",
    "adaptive_n60_mu0.03_k16": "TALON root 16",
}


def label(cfg):
    if cfg in LABELS:
        return LABELS[cfg]
    if cfg.startswith("adaptive_"):
        rest = cfg.split("_k", 1)[1]
        k, _, extra = rest.partition("_")
        return f"TALON root {k}" + (f", cap {extra.split('_')[0][1:]}" if extra.startswith("d") else "") + \
            (" + fill" if "fill" in extra else "")
    if cfg.startswith("eagle3_s"):
        s, k = cfg.split("_")[1:3]
        return f"EAGLE-3 {int(s[1:]) - 1}×{k[1:]}"
    if cfg.startswith("eagle3_tmpl_") or cfg.startswith("tmpl_"):
        return cfg.replace("eagle3_", "").replace("tmpl_", "template ")
    return cfg


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "_meta" not in d:
                yield d


def mean_widths(path):
    acc, calls = [], 0
    for r in rows(path):
        if "mean_layer_nodes" not in r:
            continue
        n = r["tree_calls"]
        calls += n
        for i, x in enumerate(r["mean_layer_nodes"]):
            acc.extend([0.0] * (i + 1 - len(acc)))
            acc[i] += x * n
    return [x / calls for x in acc] if calls else []


def find(folder, cfg, bench):
    hits = glob.glob(os.path.join(folder, cfg, f"{bench}__*__T0.jsonl"))
    return next((h for h in hits if "__ar__" not in h), None)


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def cmd_widths(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    panels = []
    for folder in args.folders:
        for path in sorted(glob.glob(os.path.join(folder, "*", "*__T0.jsonl"))):
            bench = os.path.basename(path).split("__")[0]
            if (folder, bench) not in panels:
                panels.append((folder, bench))
    configs = args.configs or sorted({os.path.basename(os.path.dirname(p))
                                      for f, _ in panels for p in glob.glob(os.path.join(f, "*", "*.jsonl"))
                                      if "__ar__" not in p})
    colour = {c: SERIES[i % len(SERIES)] for i, c in enumerate(configs)}
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False,
                             facecolor=SURFACE, sharey=True)
    table = []
    for ax, (folder, bench) in zip(axes[0], panels):
        style(ax, bench)
        for cfg in configs:
            path = find(folder, cfg, bench)
            w = mean_widths(path) if path else []
            if len(w) < 2:
                continue
            depths, widths = list(range(1, len(w))), w[1:]          # drop the root
            ax.plot(depths, widths, color=colour[cfg], linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=label(cfg))
            table += [{"folder": os.path.basename(folder.rstrip("/")), "bench": bench, "config": cfg,
                       "depth": d, "mean_width": round(x, 3)} for d, x in zip(depths, widths)]
        ax.set_xlabel("depth", color=INK_2, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0][0].set_ylabel("mean nodes at this depth", color=INK_2, fontsize=9)
    handles, labs = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labs:
                handles.append(hh)
                labs.append(ll)
    fig.legend(handles, labs, loc="upper center", ncol=min(4, len(labs)), frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    with open(os.path.splitext(args.out)[0] + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "bench", "config", "depth", "mean_width"])
        w.writeheader()
        w.writerows(table)
    print(f"wrote {args.out} and its .csv ({len(table)} points)")


def layout(parents, depths):
    """x for every node: leaves get consecutive slots, a parent sits over its children."""
    kids = {i: [] for i in range(len(parents))}
    for i, p in enumerate(parents):
        if p >= 0:
            kids[p].append(i)
    x, nxt = {}, [0]

    def place(i):
        if not kids[i]:
            x[i] = nxt[0]
            nxt[0] += 1
        else:
            for c in kids[i]:
                place(c)
            x[i] = sum(x[c] for c in kids[i]) / len(kids[i])
    place(0)
    return x


def cmd_trees(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = args.configs or sorted(os.path.basename(os.path.dirname(p))
                                     for p in glob.glob(os.path.join(args.folder, "*", f"{args.bench}__*.jsonl"))
                                     if "__ar__" not in p)
    trees = []
    for cfg in configs:
        path = find(args.folder, cfg, args.bench)
        if not path:
            continue
        for i, r in enumerate(rows(path)):
            if i == args.turn and r.get("sample_trees"):
                t = r["sample_trees"][min(args.step, len(r["sample_trees"]) - 1)]
                trees.append((cfg, t))
                break
    if not trees:
        raise SystemExit("no sample_trees found: rerun with --save-trees N")
    fig, axes = plt.subplots(len(trees), 1, figsize=(10, 2.2 * len(trees)), squeeze=False, facecolor=SURFACE)
    for ax, (cfg, t) in zip(axes[:, 0], trees):
        parents, depths = t["parents"], t["depths"]
        x = layout(parents, depths)
        span = max(x.values()) or 1
        for i, p in enumerate(parents):
            if p >= 0:
                ax.plot([x[p] / span, x[i] / span], [-depths[p], -depths[i]], color="#b9b8b2", linewidth=1, zorder=1)
        ax.scatter([x[i] / span for i in x], [-depths[i] for i in x], s=14, color=SERIES[0],
                   edgecolors=SURFACE, linewidths=0.8, zorder=2)
        ax.set_facecolor(SURFACE)
        ax.set_title(f"{label(cfg)}: {len(parents)} nodes, depth {max(depths)}", color=INK, fontsize=10, loc="left")
        ax.set_yticks([-d for d in range(max(depths) + 1)], [str(d) for d in range(max(depths) + 1)])
        ax.set_xticks([])
        ax.tick_params(colors=INK_2, labelsize=8)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.set_ylabel("depth", color=INK_2, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.out} ({len(trees)} trees)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("widths")
    w.add_argument("folders", nargs="+")
    w.add_argument("--configs", nargs="*")
    w.add_argument("--out", required=True)
    t = sub.add_parser("trees")
    t.add_argument("folder")
    t.add_argument("--bench", required=True)
    t.add_argument("--configs", nargs="*")
    t.add_argument("--turn", type=int, default=0, help="which turn (row) of the .jsonl")
    t.add_argument("--step", type=int, default=0, help="which saved tree of that turn")
    t.add_argument("--out", required=True)
    args = p.parse_args()
    cmd_widths(args) if args.cmd == "widths" else cmd_trees(args)


if __name__ == "__main__":
    main()
