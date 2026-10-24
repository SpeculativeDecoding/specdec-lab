#!/usr/bin/env python
"""Path A: sweep tree configurations x benches x batch sizes on SGLang.

One server launch per (algorithm, tree); each launch serves every bench and batch size.

  python scripts/run_sglang_sweep.py --config configs/sweep_eagle3_llama31.yaml \
      --data-dir ~/EAGLE/eagle/data --out-dir results/sglang

  # quick smoke test: one tree, one bench, 5 questions, batch 1
  python scripts/run_sglang_sweep.py --config configs/smoke.yaml \
      --data-dir ~/EAGLE/eagle/data --out-dir results/smoke
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import yaml  # noqa: E402

from specdec.data import BENCHES, load_questions, select  # noqa: E402
from specdec.sglang_client import flush_cache, run_load, server_step_times, write_run  # noqa: E402
from specdec.sglang_launcher import Server, ServerSpec  # noqa: E402
from specdec.tree_config import PLAN_GRID, TreeConfig, from_dict  # noqa: E402


def expand_servers(cfg):
    common = cfg.get("server", {})
    specs = []
    for entry in cfg["servers"]:
        mode = entry["mode"]
        base = dict(common)
        base.update({k: v for k, v in entry.items() if k not in ("mode", "trees", "plan_budgets", "mu")})
        if mode == "talon":  # TALON-S (specdec/sglang_talon_plugin.py) on each listed tree
            for t in entry.get("trees", []):
                specs.append(ServerSpec(mode="talon", tree=from_dict(t).validate(),
                                        talon_mu=float(entry.get("mu", 0.03)), **base))
        elif mode == "spec":
            trees = [from_dict(t) for t in entry.get("trees", [])]
            for b in entry.get("plan_budgets", []):
                trees += PLAN_GRID[int(b)]
            for t in trees:
                specs.append(ServerSpec(mode="spec", tree=t.validate(), **base))
        elif mode == "adaptive":
            st = base.pop("adaptive_start", None)
            s = ServerSpec(mode="adaptive", **base)
            if st:
                s.adaptive_start = from_dict(st)
            specs.append(s)
        else:
            specs.append(ServerSpec(mode=mode, **base))
    return specs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dry-run", action="store_true", help="print server commands and exit")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    run = cfg.get("run", {})
    benches = run.get("benches", list(BENCHES))
    batch_sizes = [int(b) for b in run.get("batch_sizes", [1, 8, 32])]
    temps = [float(t) for t in run.get("temperatures", [0.0])]
    subset = run.get("subset", "all")
    limit = int(run.get("limit", 0))
    max_turns = int(run.get("max_turns", 0))
    max_new = int(run.get("max_new_tokens", 1024))
    repeat_for_bs = {int(k): int(v) for k, v in run.get("repeat_prompts_for_batch", {32: 2}).items()}
    warmup_q = int(run.get("warmup_questions", 4))

    specs = expand_servers(cfg)
    if max(batch_sizes) > min(s.max_running_requests for s in specs):
        sys.exit("max batch size exceeds max_running_requests; raise it in the config")
    for s in specs:
        print(s.name, ":", " ".join(s.command()))
    if args.dry_run:
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(specs[0].model_path)
    questions = {b: select(load_questions(args.data_dir, b), subset, limit) for b in benches}

    os.makedirs(args.out_dir, exist_ok=True)
    for s in specs:
        todo = [(b, bs, T) for b in benches for bs in batch_sizes for T in temps
                if not (args.skip_existing and os.path.exists(
                    os.path.join(args.out_dir, s.name, f"{b}__bs{bs}__T{T:g}.jsonl")))]
        if not todo:
            continue
        log = os.path.join(args.out_dir, s.name, "server.log")
        with Server(s, log) as srv:
            # warm-up: CUDA graphs and allocator, discarded
            run_load(srv.base_url, tok, questions[benches[0]][:warmup_q], min(4, max(batch_sizes)),
                     max_new_tokens=128, max_turns=1)
            srv.verify_talon()  # TALON-S servers: refuse to time a run the hook did not touch
            for bench, bs, T in todo:
                qs = questions[bench] * repeat_for_bs.get(bs, 1)
                flush_cache(srv.base_url)
                t0 = time.time()
                res = run_load(srv.base_url, tok, qs, bs, max_new_tokens=max_new,
                               temperature=T, max_turns=max_turns)
                res["summary"]["server_step_time"] = server_step_times(srv.base_url, bs)
                meta = {"path": "sglang", "server": s.to_dict(), "bench": bench, "batch_size": bs,
                        "temperature": T, "subset": subset, "limit": limit, "max_turns": max_turns,
                        "max_new_tokens": max_new, "prompt_repeats": repeat_for_bs.get(bs, 1),
                        "wall_clock_start": t0}
                out = os.path.join(args.out_dir, s.name, f"{bench}__bs{bs}__T{T:g}.jsonl")
                write_run(out, meta, res)
                sm = res["summary"]
                print(f"[{s.name}] {bench} bs={bs} T={T:g}: "
                      f"{sm['throughput_all']:.1f} tok/s (full-occ {sm['throughput_full_occupancy']}), "
                      f"tau={sm.get('tau')}, errors={sm['n_errors']}", flush=True)
                if sm["n_errors"]:
                    print(json.dumps(sm["errors"], indent=1))


if __name__ == "__main__":
    main()
