"""TALON-S: TALON's tree rules inside SGLang's EAGLE-3 drafter (Path A, any batch size).

SGLang v0.5.22 builds the draft tree in `eagle_worker_v2.draft_forward`, one layer per step:

  select_top_k_tokens(i, topk_p, topk_index, hidden_states, scores, topk)
      -> input_ids, hidden_states, beam_scores, (tree_scores, tokens, parents)

`beam_scores` (b, topk) are the cumulative probabilities of the `topk` nodes expanded next;
`tree_scores` are the cumulative probabilities of every candidate of the layer, (b, 1, topk) at
i = 0 and (b, topk, topk) after. `organize_draft_results` then verifies the global top
(num_draft_tokens - 1) candidates by `tree_scores`. That is EAGLE-2's rule.

This plugin wraps `select_top_k_tokens` (an AROUND hook through SGLang's own HookRegistry,
applied in the scheduler process that runs the drafter) and rewrites only the scores, so the
captured CUDA graph keeps its fixed shapes. TALON's Algorithm 2 (Liu et al., arXiv 2601.07353):

  * Layer 0, robust init: every one of the top-K root children is kept (needs topk == K).
  * Layer d >= 1, confidence gate: a candidate is kept iff its parent was kept and
    p(u) >= mu * m_d, m_d = the best candidate of the layer.
  * Only kept nodes are expanded: a gated beam gets a negative score, so its children rank
    below every child of a kept beam in the next layer's top-k and can never pass the gate.
  * Budget check at every layer: kept nodes are verified layer by layer (earlier layers first,
    best first within a layer) until num_draft_tokens is reached. Implemented as the selection
    key (num_steps - d) + p(u) / m_d for kept nodes; gated nodes get |p(u)| / 2 < 1, so they
    are verified only as fillers when fewer kept nodes than the budget exist.

Every key is >= its parent's key, so the verified set is always a valid tree.

Known limits of the "S" variant (fixed shapes): at most `topk` nodes are expanded per layer,
the tree has at most `steps` layers, and the drafter always runs all `steps` passes. Full
TALON (variable width, depth and number of draft passes) runs only in Path B, batch size 1.

Enabled per server by SPECDEC_TALON=1 (with SPECDEC_TALON_MU, SPECDEC_TALON_NUM_STEPS,
SPECDEC_TALON_ROOT_K, SPECDEC_TALON_OUT_DIR). Registered as the `sglang.srt.plugins` entry
point `talon` (chtc/job_sglang.sh writes the dist-info). Each process that registers the hook,
and each first call of it, appends a line to <SPECDEC_TALON_OUT_DIR>/talon_marker.jsonl,
which the launcher checks before any timed run.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

TARGET = "sglang.srt.speculative.eagle_worker_v2.select_top_k_tokens"
MARKER = "talon_marker.jsonl"
_TINY = 1e-30


@dataclass(frozen=True)
class TalonSConfig:
    mu: float = 0.03       # confidence threshold, 0 < mu <= 1
    num_steps: int = 9     # SGLang --speculative-num-steps (tree layers)
    root_k: int = 10       # forced root width K; must equal --speculative-eagle-topk

    @classmethod
    def from_env(cls):
        if os.environ.get("SPECDEC_TALON") != "1":
            return None
        return cls(mu=float(os.environ.get("SPECDEC_TALON_MU", "0.03")),
                   num_steps=int(os.environ.get("SPECDEC_TALON_NUM_STEPS", "9")),
                   root_k=int(os.environ.get("SPECDEC_TALON_ROOT_K", "10"))).validate()

    def validate(self):
        if not 0.0 < self.mu <= 1.0:
            raise ValueError(f"mu must be in (0, 1], got {self.mu}")
        if self.num_steps < 1 or self.root_k < 1:
            raise ValueError("num_steps and root_k must be >= 1")
        return self


def talon_rescore(i: int, tree_scores, beam_scores, num_steps: int, mu: float):
    """TALON selection keys for layer i, and the beam scores to expand with.

    Pure tensor ops (no host sync, no data-dependent shapes), so it is CUDA-graph safe.
    tree_scores: (b, 1, topk) at i == 0, else (b, topk, topk); beam_scores: (b, topk).
    Returns (keys as float32 with tree_scores' shape, beam_scores with its dtype).
    """
    import torch

    s = tree_scores.float()
    b = s.shape[0]
    if i == 0:  # robust init: all top-K root children, best first
        m = s.reshape(b, -1).amax(-1).clamp_min(_TINY).view(b, 1, 1)
        return s / m + float(num_steps), beam_scores
    pos = s > 0  # children of kept beams (gated beams carry negative scores)
    m = torch.where(pos, s, torch.zeros_like(s)).reshape(b, -1).amax(-1).clamp_min(_TINY)
    kept = pos & (s >= mu * m.view(b, 1, 1))
    keys = torch.where(kept, s / m.view(b, 1, 1) + float(num_steps - i), s.abs() * 0.5)
    bs = beam_scores
    beam_kept = (bs > 0) & (bs.float() >= mu * m.view(b, 1))
    return keys, torch.where(beam_kept, bs, -bs.abs())


def _write_marker(event: str, cfg: TalonSConfig, **extra):
    path = os.path.join(os.environ.get("SPECDEC_TALON_OUT_DIR", "."), MARKER)
    rec = {"event": event, "pid": os.getpid(), "time": time.time(), **asdict(cfg), **extra}
    try:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as e:  # never take the server down over a marker
        logger.warning("TALON-S: could not write %s: %s", path, e)


def make_hook(cfg: TalonSConfig):
    called = {"done": False}

    def around(original_fn, *args, **kwargs):
        input_ids, hidden_states, beam_scores, tree_info = original_fn(*args, **kwargs)
        i = args[0] if args else kwargs["i"]
        topk = args[5] if len(args) > 5 else kwargs.get("topk")
        if topk is not None and topk != cfg.root_k:
            raise ValueError(f"TALON-S needs --speculative-eagle-topk == root_k ({cfg.root_k}), got {topk}")
        keys, beam_scores = talon_rescore(i, tree_info[0], beam_scores, cfg.num_steps, cfg.mu)
        if not called["done"]:  # first call happens during CUDA-graph capture, before serving
            called["done"] = True
            _write_marker("called", cfg, layer=int(i))
        return input_ids, hidden_states, beam_scores, (keys, tree_info[1], tree_info[2])

    return around


def plugin():
    """`sglang.srt.plugins` entry point; SGLang calls it in every process it starts."""
    cfg = TalonSConfig.from_env()
    if cfg is None:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, make_hook(cfg), HookType.AROUND)
    _write_marker("registered", cfg)
    logger.info("TALON-S hook registered on %s: %s", TARGET, cfg)
