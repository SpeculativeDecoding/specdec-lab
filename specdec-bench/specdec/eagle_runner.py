"""Path B: EAGLE-3 speculative decoding with the official EAGLE repository (batch size 1).

Every call below was checked against SafeAILab/EAGLE @ cb7e0841 (Feb 2026):
  EaModel.from_pretrained(use_eagle3, base_model_path, ea_model_path,
                          total_token=60, depth=7, top_k=10, threshold=1.0, **hf_kwargs)
  EaModel.eagenerate(input_ids, temperature=0.0, top_p=0.0, top_k=0.0,
                     max_new_tokens=512, max_length=2048, log=False, is_llama3=False)
      -> (output_ids, new_token, idx) when log=True; verify passes = idx + 1
  EaModel.naivegenerate(...) same signature; plain autoregressive decoding with the
      same base model and the same KV-cache implementation (official AR baseline).

Two traps in the official code that this runner avoids:
  1. The KV cache is allocated ONCE, on the first eagenerate/naivegenerate call, with
     that call's max_length, and reused afterwards. We always pass the same
     max_length (default 4096) so long prompts (CNN/DM) never overflow it.
  2. The official eval script never passes max_new_tokens, so it silently caps at
     512. We pass it explicitly to both speculative and baseline generation.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, Iterable, List, Optional

from .data import encode_messages, initial_messages, stop_token_ids, truncate_at_stop
from .adaptive import AdaptiveConfig, TemplateConfig, enable_adaptive, enable_template
from .tree_config import TreeConfig

DEFAULT_BASE = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_EAGLE3 = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


def load_eagle_model(tree: Optional[TreeConfig], base_model_path: str = DEFAULT_BASE,
                     ea_model_path: str = DEFAULT_EAGLE3, use_eagle3: bool = True,
                     dtype: str = "float16", adaptive: Optional[AdaptiveConfig] = None,
                     instrument: bool = True, save_trees: int = 0):
    """Load target + draft exactly as the official llama3 eval script does.

    tree:  static EAGLE tree (steps/topk/budget). Ignored when `adaptive` is given.
    save_trees: keep the first `save_trees` draft trees of every turn (parents and depths).
    adaptive: an AdaptiveConfig (TALON) or a TemplateConfig (fixed widths per depth) that builds
           every draft tree (EAGLE-3 drafters only). The model is
           loaded with total_token = adaptive.budget so EAGLE's length guard matches.
    """
    import torch  # imported lazily so the package imports without torch
    from eagle.model.ea_model import EaModel

    if adaptive is not None:
        adaptive.validate()
        if not use_eagle3:
            raise ValueError("adaptive tree is implemented for EAGLE-3 draft layers only (as in the paper)")
        # depth/top_k only size EAGLE's unused static buffers when adaptive tree builds the tree
        kwargs = {"total_token": adaptive.budget, "depth": 5, "top_k": adaptive.root_k}
        expect_total = adaptive.budget
    else:
        if tree is None:
            raise ValueError("give a TreeConfig or an AdaptiveConfig")
        tree.validate()
        kwargs = tree.eagle_kwargs()   # total_token, depth, top_k
        expect_total = tree.eagle_total_token
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    model = EaModel.from_pretrained(
        use_eagle3=use_eagle3,
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        **kwargs,
    )
    model.eval()
    # Defensive check: the draft layer stores total_token - 1 (cnets.Model.__init__).
    got = getattr(model.ea_layer, "total_tokens", None)
    if got is not None and got != expect_total - 1:
        raise RuntimeError(f"ea_layer.total_tokens={got}, expected {expect_total - 1}")
    if isinstance(adaptive, TemplateConfig):
        enable_template(model, adaptive)
    elif adaptive is not None:
        enable_adaptive(model, adaptive)
    if instrument:
        instrument_drafting(model, static_depth=None if adaptive else tree.eagle_depth,
                            save_trees=save_trees)
    return model


def tree_parents(mask, pos):
    """Parent of every node of an EAGLE draft tree (index 0 = root, parent -1), from its
    attention mask (row i attends to i's ancestors) and depths (tree_position_ids)."""
    m = mask.reshape(mask.shape[-2], mask.shape[-1]).tolist() if hasattr(mask, "reshape") else mask
    pos = [int(x) for x in (pos.flatten().tolist() if hasattr(pos, "flatten") else pos)]
    parents = [-1] * len(pos)
    for i in range(1, len(pos)):
        for j in range(len(pos)):
            if m[i][j] and pos[j] == pos[i] - 1:
                parents[i] = j
                break
    return parents, pos


def instrument_drafting(model, static_depth: Optional[int], save_trees: int = 0):
    """Count draft forward passes and record realized tree depth per topK_genrate call.

    Static EAGLE runs exactly 1 + depth draft passes per call (context pass + `depth`
    loop passes in cnets.Model.topK_genrate). adaptive tree reports its own count. adaptive tree's
    draft efficiency is delta = draft passes / verify passes (Section 5.3).
    """
    layer = model.ea_layer
    inner = layer.topK_genrate
    stats = {"calls": 0, "forwards": 0, "depth_sum": 0, "nodes_sum": 0, "layer_nodes": [], "trees": []}
    model._draft_stats = stats

    def wrapped(*args, **kwargs):
        out = inner(*args, **kwargs)
        stats["calls"] += 1
        adaptive = getattr(layer, "adaptive", None)
        if adaptive is not None:
            stats["forwards"] += adaptive.last["forwards"]
            stats["depth_sum"] += adaptive.last["depth"]
            stats["nodes_sum"] += adaptive.last["nodes"]
        else:
            stats["forwards"] += 1 + static_depth
            stats["depth_sum"] += int(out[3].max())          # tree_position_ids
            stats["nodes_sum"] += int(out[0].shape[1])       # draft_tokens incl. root
        # nodes per tree depth (index 0 = root), summed over calls: the tree's width by layer
        pos = out[3]
        pos = pos.flatten().tolist() if hasattr(pos, "flatten") else list(pos)
        for d in pos:
            d = int(d)
            if d >= len(stats["layer_nodes"]):
                stats["layer_nodes"].extend([0] * (d + 1 - len(stats["layer_nodes"])))
            stats["layer_nodes"][d] += 1
        if len(stats["trees"]) < save_trees:
            parents, depths = tree_parents(out[2], out[3])
            stats["trees"].append({"parents": parents, "depths": depths})
        return out

    layer.topK_genrate = wrapped
    return stats


def _cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class EagleRunner:
    """Runs one benchmark at batch size 1 and writes one JSON line per turn."""

    def __init__(self, model, tokenizer=None, *, max_new_tokens: int = 1024,
                 max_length: int = 4096, temperature: float = 0.0,
                 sync: Optional[Callable[[], None]] = None,
                 to_device: Optional[Callable] = None,
                 seed_fn: Optional[Callable[[int], None]] = None):
        self.model = model
        self.tok = tokenizer if tokenizer is not None else model.get_tokenizer()
        self.max_new_tokens = int(max_new_tokens)
        self.max_length = int(max_length)
        self.temperature = float(temperature)
        self.sync = sync or _cuda_sync
        self.to_device = to_device or self._default_to_device
        self.seed_fn = seed_fn or self._default_seed
        self.stops = stop_token_ids(self.tok)

    @staticmethod
    def _default_to_device(ids: List[int]):
        import torch
        t = torch.as_tensor([ids])
        return t.cuda() if torch.cuda.is_available() else t

    @staticmethod
    def _default_seed(i: int):
        import torch
        torch.manual_seed(i)

    # ------------------------------------------------------------------
    def generate(self, input_ids: List[int], mode: str) -> Dict:
        if mode not in ("spec", "ar"):
            raise ValueError("mode must be 'spec' or 'ar'")
        if len(input_ids) + self.max_new_tokens + 80 > self.max_length:
            # eagenerate stops early when input_ids exceeds max_length - total_tokens - 10;
            # flag it rather than silently truncating.
            truncated_risk = True
        else:
            truncated_risk = False
        fn = self.model.eagenerate if mode == "spec" else self.model.naivegenerate
        stats = getattr(self.model, "_draft_stats", None)
        if stats is not None:
            for k in stats:
                stats[k] = [] if isinstance(stats[k], list) else 0
        x = self.to_device(input_ids)
        self.sync()
        t0 = time.perf_counter()
        output_ids, new_token, idx = fn(
            x,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            max_length=self.max_length,
            log=True,
            is_llama3=True,
        )
        self.sync()
        wall = time.perf_counter() - t0
        gen = output_ids[0][len(input_ids):]
        gen = gen.tolist() if hasattr(gen, "tolist") else list(gen)
        kept = truncate_at_stop(gen, self.stops)
        text = self.tok.decode(kept, spaces_between_special_tokens=False).strip()
        rec = {
            "output_ids": kept,
            "text": text,
            "output_tokens": len(kept),
            "new_tokens": int(new_token),
            "verify_steps": int(idx) + 1,
            "tau": int(new_token) / (int(idx) + 1),
            "wall_time": wall,
            "prompt_tokens": len(input_ids),
            "hit_stop": len(kept) < len(gen),
            "length_risk": truncated_risk,
        }
        if stats is not None and mode == "spec" and stats["calls"]:
            # every topK_genrate call builds one tree; the first one follows prefill
            rec["draft_forwards"] = stats["forwards"]
            rec["tree_calls"] = stats["calls"]
            rec["delta"] = stats["forwards"] / rec["verify_steps"]
            rec["mean_tree_depth"] = stats["depth_sum"] / stats["calls"]
            rec["mean_tree_nodes"] = stats["nodes_sum"] / stats["calls"]
            # mean nodes at each depth of the verified tree, [root, depth 1, depth 2, ...]
            rec["mean_layer_nodes"] = [n / stats["calls"] for n in stats["layer_nodes"]]
            if stats["trees"]:
                rec["sample_trees"] = list(stats["trees"])
        return rec

    def run_question(self, q: Dict, mode: str, seed: int = 0) -> List[Dict]:
        """All turns of one question, feeding each answer back like the official script."""
        self.seed_fn(seed)
        messages = initial_messages()
        rows = []
        for j, turn in enumerate(q["turns"]):
            messages.append({"role": "user", "content": turn})
            ids = encode_messages(self.tok, messages)
            r = self.generate(ids, mode)
            messages.append({"role": "assistant", "content": r["text"]})
            r.update(bench=q["bench"], question_id=q["question_id"], turn=j,
                     qid_turn=f'{q["bench"]}:{q["question_id"]}:{j}', mode=mode)
            rows.append(r)
        return rows

    def run(self, questions: Iterable[Dict], mode: str, out_path: str,
            meta: Optional[Dict] = None, warmup: int = 3, max_turns: int = 0,
            keep_ids: bool = True, resume: bool = True) -> List[Dict]:
        """Run every question and write one JSON line per turn.

        Checkpointing: rows are appended to `out_path + ".partial"` and flushed after
        every question; the file is renamed to `out_path` only when all questions are
        done. With resume=True, an existing .partial written with the *same* meta
        (same model, tree, GPU name, library versions, ...) is reloaded: finished
        questions are kept and only the rest are run. A question is kept only if all
        its turns are present, and a line cut off by a crash is dropped. A .partial
        with different meta is discarded, so timings are never mixed across setups.
        """
        questions = list(questions)
        if max_turns:
            questions = [dict(q, turns=q["turns"][:max_turns]) for q in questions]
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        tmp_path = out_path + ".partial"
        rows: List[Dict] = []
        done = set()
        if resume and os.path.exists(tmp_path):
            rows, done = load_checkpoint(tmp_path, meta, questions)
            if done:
                print(f"[resume] {out_path}: {len(done)}/{len(questions)} questions already done",
                      flush=True)
        todo = [q for q in questions if _qkey(q) not in done]
        if warmup and todo:
            for _ in range(warmup):
                self.run_question(todo[0], mode)
        # Rewrite the checkpoint with only the complete questions (atomic), then append.
        _write_rows(tmp_path, meta, rows, keep_ids)
        with open(tmp_path, "a", encoding="utf-8") as f:
            for q in todo:
                qrows = self.run_question(q, mode)
                for r in qrows:
                    rec = dict(r)
                    if not keep_ids:
                        rec.pop("output_ids", None)
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())          # survives a hard kill of the VM / Colab runtime
                rows.extend(qrows)
        os.replace(tmp_path, out_path)
        return rows


def _qkey(q: Dict):
    return (q["bench"], q["question_id"])


def _norm(x):
    return json.loads(json.dumps(x))


def _write_rows(path: str, meta: Optional[Dict], rows: List[Dict], keep_ids: bool = True):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        if meta is not None:
            f.write(json.dumps({"_meta": meta}) + "\n")
        for r in rows:
            rec = dict(r)
            if not keep_ids:
                rec.pop("output_ids", None)
            f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path: str, meta: Optional[Dict], questions: List[Dict]):
    """Complete questions from a .partial file -> (rows in file order, set of keys).

    Returns nothing reusable if the meta differs or the file is unreadable.
    """
    expected = {_qkey(q): len(q["turns"]) for q in questions}
    file_meta, parsed = None, []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.endswith("\n"):
                    break                        # last line cut off mid-write
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    break
                if i == 0 and "_meta" in d:
                    file_meta = d["_meta"]
                    continue
                parsed.append(d)
    except OSError:
        return [], set()
    if _norm(file_meta) != _norm(meta):
        print(f"[resume] {path}: written with different settings, starting over", flush=True)
        return [], set()
    by_q: Dict = {}
    for d in parsed:
        by_q.setdefault((d.get("bench"), d.get("question_id")), []).append(d)
    done = set()
    for k, rs in by_q.items():
        if k in expected and [r.get("turn") for r in rs] == list(range(expected[k])):
            done.add(k)
    rows = [d for d in parsed if (d.get("bench"), d.get("question_id")) in done]
    return rows, done


def read_jsonl(path: str):
    meta, rows = None, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "_meta" in d:
                meta = d["_meta"]
            else:
                rows.append(d)
    return meta, rows
