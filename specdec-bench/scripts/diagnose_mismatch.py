%%writefile /content/specdec-bench/scripts/diagnose_mismatch.py
#!/usr/bin/env python
"""Is each spec-vs-AR mismatch an fp16 near-tie or a real bug?

For every answer where the two files differ, rebuild the exact prompt that both runs saw,
append the shared output up to the first differing token, run the target model once,
and look at the logits of the two tokens the runs chose there.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec.data import encode_messages, initial_messages, load_questions  # noqa: E402
from specdec.eagle_runner import DEFAULT_BASE, read_jsonl  # noqa: E402

NEAR_TIE = 0.25      # logit gap below which fp16 reordering can plausibly flip the arg-max


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def prompt_ids(tok, q, turn, rows_by_turn):
    """Prompt for `turn`, feeding back this file's own earlier answers (as the runner did)."""
    msgs = initial_messages()
    for j in range(turn + 1):
        msgs.append({"role": "user", "content": q["turns"][j]})
        if j < turn:
            msgs.append({"role": "assistant", "content": rows_by_turn[j]["text"]})
    return encode_messages(tok, msgs)


def classify(logits, spec_tok, ar_tok):
    import numpy as np
    x = np.asarray(logits, dtype=np.float64)
    order = np.argsort(-x)
    rank = {int(t): int(np.where(order == t)[0][0]) + 1 for t in (spec_tok, ar_tok)}
    p = np.exp(x - x.max())
    p /= p.sum()
    gap = abs(float(x[spec_tok] - x[ar_tok]))
    tie = gap < NEAR_TIE and max(rank.values()) <= 2
    return {"gap": gap, "rank_spec": rank[spec_tok], "rank_ar": rank[ar_tok],
            "p_spec": float(p[spec_tok]), "p_ar": float(p[ar_tok]),
            "verdict": "fp16 near-tie" if tie else "SUSPICIOUS"}


def analyse(spec_rows, ar_rows, questions, tok, logits_fn):
    S = {r["qid_turn"]: r for r in spec_rows}
    A = {r["qid_turn"]: r for r in ar_rows}
    qmap = {(q["bench"], str(q["question_id"])): q for q in questions}
    out = []
    for k in sorted(set(S) & set(A), key=str):
        s, a = S[k], A[k]
        if s["output_ids"] == a["output_ids"]:
            continue
        bench, qid, turn = k.rsplit(":", 2)
        turn = int(turn)
        earlier = [f"{bench}:{qid}:{j}" for j in range(turn)]
        if any(S[e]["output_ids"] != A[e]["output_ids"] for e in earlier if e in S and e in A):
            out.append({"key": k, "verdict": "inherited (earlier turn differed)"})
            continue
        q = qmap[(bench, qid)]
        ids = prompt_ids(tok, q, turn, {j: S[f"{bench}:{qid}:{j}"] for j in range(turn)})
        if len(ids) != s["prompt_tokens"]:
            out.append({"key": k, "verdict": f"prompt rebuild mismatch ({len(ids)} vs {s['prompt_tokens']})"})
            continue
        pos = first_divergence(s["output_ids"], a["output_ids"])
        if pos >= min(len(s["output_ids"]), len(a["output_ids"])):
            out.append({"key": k, "pos": pos, "verdict": "one output stops earlier (stop-token near-tie)"})
            continue
        st, at = s["output_ids"][pos], a["output_ids"][pos]
        r = classify(logits_fn(ids + s["output_ids"][:pos]), st, at)
        r.update(key=k, pos=pos, spec_tok=tok.decode([st]), ar_tok=tok.decode([at]))
        out.append(r)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spec_jsonl")
    p.add_argument("ar_jsonl")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    a = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _, spec_rows = read_jsonl(a.spec_jsonl)
    _, ar_rows = read_jsonl(a.ar_jsonl)
    benches = sorted({r["bench"] for r in spec_rows})
    questions = [q for b in benches for q in load_questions(a.data_dir, b)]
    tok = AutoTokenizer.from_pretrained(a.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        a.base_model, torch_dtype=getattr(torch, a.dtype), device_map="auto").eval()

    @torch.no_grad()
    def logits_fn(ids):
        x = torch.tensor([ids], device=model.device)
        return model(x).logits[0, -1].float().cpu().numpy()

    rows = analyse(spec_rows, ar_rows, questions, tok, logits_fn)
    for r in rows:
        if "gap" in r:
            print(f"{r['key']:>16s} @tok {r['pos']:4d}: spec {r['spec_tok']!r} (rank {r['rank_spec']}, p={r['p_spec']:.3f}) "
                  f"vs plain {r['ar_tok']!r} (rank {r['rank_ar']}, p={r['p_ar']:.3f}), "
                  f"logit gap {r['gap']:.3f} -> {r['verdict']}")
        else:
            print(f"{r['key']:>16s}: {r['verdict']}")
    sus = [r for r in rows if r.get("verdict") == "SUSPICIOUS"]
    print(f"\n{len(rows)} differing turns: "
          f"{sum(r.get('verdict') == 'fp16 near-tie' for r in rows)} near-ties, "
          f"{sum(str(r.get('verdict')).startswith('inherited') for r in rows)} inherited, "
          f"{len(sus)} suspicious")


if __name__ == "__main__":
    main()