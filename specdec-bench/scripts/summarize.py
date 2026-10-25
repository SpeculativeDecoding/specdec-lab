#!/usr/bin/env python
"""Collect every result file under a directory into one CSV, with speedups vs plain decoding.

  python scripts/summarize.py results/ --out results/summary.csv

Speedup pairs each run with the AR run of the same path, bench, batch size and
temperature. At batch size 1 a paired bootstrap 95% CI over prompts is added.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec.metrics import paired_bootstrap_speedup, pooled_tau, throughput  # noqa: E402


def load(path):
    meta, summary, rows = None, None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "_meta" in d:
                meta, summary = d["_meta"], d.get("_summary")
            else:
                rows.append(d)
    return meta, summary, rows


def describe(meta):
    if meta["path"] == "sglang":
        srv = meta["server"]
        return srv["name"], srv["mode"], (srv["tree"] or {})
    alg = "eagle3" if meta.get("use_eagle3", True) else "eagle1"
    if meta["mode"] != "spec":
        return "ar", meta["mode"], {}
    if meta.get("template"):
        t = meta["template"]
        return f"{alg}_{t['name']}", "template", {"num_draft_tokens": t["budget"],
                                                  "steps": len(t["widths"]), "topk": t["branch"]}
    adaptive = meta.get("adaptive")
    if adaptive is None:
        adaptive = next((value for value in meta.values()
                         if isinstance(value, dict) and {"budget", "mu", "root_k"} <= value.keys()), None)
    if adaptive:
        name = f"adaptive_n{adaptive['budget']}_mu{adaptive['mu']:g}_k{adaptive['root_k']}"
        return f"{alg}_{name}", "adaptive", {"num_draft_tokens": adaptive["budget"], "mu": adaptive["mu"],
                                              "root_k": adaptive["root_k"]}
    tree = meta.get("tree") or {}
    return f"{alg}_{tree['name']}", "spec", tree


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()

    runs = []
    for path in sorted(glob.glob(os.path.join(args.root, "**", "*.jsonl"), recursive=True)):
        meta, summary, rows = load(path)
        if not meta or not rows:
            continue
        name, mode, tree = describe(meta)
        runs.append(dict(path=path, meta=meta, summary=summary or {}, rows=rows,
                         name=name, mode=mode, tree=tree))

    ar_index = {}
    for r in runs:
        if r["mode"] == "ar":
            m = r["meta"]
            ar_index[(m["path"], m["bench"], m.get("batch_size", 1), m["temperature"])] = r

    out_rows = []
    for r in runs:
        m, s = r["meta"], r["summary"]
        bs = m.get("batch_size", 1)
        key = (m["path"], m["bench"], bs, m["temperature"])
        if m["path"] == "sglang":
            tps = s.get("throughput_full_occupancy") or s.get("throughput_all")
        else:
            tps = throughput(r["rows"])
        spec_rows = [x for x in r["rows"] if "verify_steps" in x]
        row = {
            "path": m["path"], "config": r["name"], "mode": r["mode"], "bench": m["bench"],
            "batch_size": bs, "temperature": m["temperature"],
            "steps": r["tree"].get("steps"), "topk": r["tree"].get("topk"),
            "budget": r["tree"].get("num_draft_tokens"),
            "n_turns": len(r["rows"]), "throughput_tok_s": round(tps, 2) if tps else None,
            "tau": round(pooled_tau(spec_rows), 4) if spec_rows and r["mode"] != "ar" else None,
            "errors": s.get("n_errors", 0),
            "mu": r["tree"].get("mu"), "root_k": r["tree"].get("root_k"),
        }
        fw = [x for x in spec_rows if "draft_forwards" in x]
        if fw:
            row["delta"] = round(sum(x["draft_forwards"] for x in fw) / sum(x["verify_steps"] for x in fw), 4)
            calls = sum(x["tree_calls"] for x in fw)
            row["mean_tree_depth"] = round(sum(x["mean_tree_depth"] * x["tree_calls"] for x in fw) / calls, 3)
        ar = ar_index.get(key)
        if ar is not None and r["mode"] != "ar":
            if m["path"] == "sglang":
                as_ = ar["summary"]
                ar_tps = as_.get("throughput_full_occupancy") or as_.get("throughput_all")
            else:
                ar_tps = throughput(ar["rows"])
            row["speedup"] = round(tps / ar_tps, 4) if tps and ar_tps else None
            if bs == 1:
                ci = paired_bootstrap_speedup(r["rows"], ar["rows"], n_boot=args.n_boot)
                row["speedup_ci_lo"], row["speedup_ci_hi"] = round(ci["lo"], 4), round(ci["hi"], 4)
        out_rows.append(row)

    cols = ["path", "config", "mode", "bench", "batch_size", "temperature", "steps", "topk",
            "budget", "mu", "root_k", "n_turns", "throughput_tok_s", "tau", "delta",
            "mean_tree_depth", "speedup", "speedup_ci_lo", "speedup_ci_hi", "errors"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in sorted(out_rows, key=lambda x: (x["path"], x["bench"], x["batch_size"], x["config"])):
            w.writerow({c: row.get(c) for c in cols})
    print(f"wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
