"""adaptive tree: budget-driven adaptive draft trees (Liu et al., arXiv 2601.07353, Algorithm 2).

Drop-in replacement for the EAGLE-3 draft-tree builder `ea_layer.topK_genrate`
(SafeAILab/EAGLE @ cb7e0841, eagle/model/cnets.py). Verification is untouched, so
the method stays lossless exactly as the paper states (Appendix B).

Algorithm 2, as implemented here
--------------------------------
  T <- {x_t}  (the root counts towards the budget N; N == EAGLE total_token)
  Layer d = 0 (robust init):  P_1 = Top-K children of the root (K = 10 in the paper).
  Layer d >= 1 (confidence gating):
      S_d = all children (v, w) of every v in P_d, scored p(u) = p(v) * P_draft(w | path)
      m_d = max_{u in S_d} p(u)
      P_{d+1} = { u in S_d : p(u) >= mu * m_d }            (mu = 0.03 in the paper)
  Budget check at EVERY layer: if |T| + |P_{d+1}| > N keep only the top (N - |T|) by p(u).
  T <- T u P_{d+1}; repeat while |T| < N. Every node of P_{d+1} is expanded next.
Each layer adds at least the anchor node, so the loop always terminates with |T| = N.

Exactness of the candidate pre-filter
-------------------------------------
Scoring the full vocabulary for every parent is wasteful. Each parent's children are
first cut to its top-M with M = min(remaining budget R, vocab). This is exact:
  * if more than R candidates pass the gate, only the global top-R survive, and the
    global top-R are always inside the union of each parent's top-R;
  * if at most R pass, every passing child of a parent is among that parent's
    highest-scoring children (the gate is monotone in the score), so it is in its top-R.
The anchor m_d is the best child of some parent, so it is in the pre-filtered set too.

Correspondence with EAGLE's static builder (kept identical on purpose)
---------------------------------------------------------------------
  * Context forward, stable_kv handling, positions (len_posi), first-layer hidden
    (last_hidden repeated), child hidden = parent's output hidden, draft->target id
    map (d2t) and the tree-attention mask update cat(mask[parents], eye(n_new)) are
    the same operations as cnets.Model.topK_genrate; only the number of nodes per
    layer is no longer fixed at top_k.
  * Scores are cumulative log-probabilities (EAGLE's convention); the gate
    p >= mu*m becomes logp >= log(mu) + log(m).
  * Outputs (draft_tokens, retrieve_indices, tree_mask, tree_position_ids) follow
    EAGLE's format, including its retrieve-index sort when sampling (T > 0).

Only EAGLE-3 draft layers (cnets.Model) are supported, which is what the paper uses.
"""
from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Dict, List, Optional


# ----------------------------------------------------------------------------- config
@dataclass(frozen=True)
class AdaptiveConfig:
    budget: int = 60        # N: verified nodes INCLUDING the root (== EAGLE total_token)
    mu: float = 0.03        # confidence threshold, 0 < mu <= 1
    root_k: int = 10        # K: width of the forced top-K root layer
    max_depth: int = 0      # optional safety cap on layers (0 = none, as in the paper)
    fill: bool = False      # with max_depth: spend budget left at the cap on the best unused
                            # candidates of the layers already built (no extra draft passes)

    def validate(self) -> "AdaptiveConfig":
        if self.budget < 2:
            raise ValueError("budget N must be >= 2 (root + at least one draft node)")
        if not (0.0 < self.mu <= 1.0):
            raise ValueError("mu must be in (0, 1]")
        if self.root_k < 1:
            raise ValueError("root_k must be >= 1")
        if self.fill and not self.max_depth:
            raise ValueError("fill only applies with a depth cap (max_depth > 0)")
        if self.max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        return self

    @property
    def name(self) -> str:
        cap = f"_d{self.max_depth}" if self.max_depth else ""   # unchanged name when uncapped
        fill = "_fill" if self.fill else ""
        return f"adaptive_n{self.budget}_mu{self.mu:g}_k{self.root_k}{cap}{fill}"

    def to_dict(self) -> Dict:
        return {"budget": self.budget, "mu": self.mu, "root_k": self.root_k,
                "max_depth": self.max_depth, "fill": self.fill, "name": self.name}


# ----------------------------------------------------------------------------- tensor ops
class TorchOps:
    """The handful of tensor operations adaptive tree needs, on torch."""

    def __init__(self):
        import torch
        self.t = torch

    def topk(self, x, k):
        r = self.t.topk(x, k, dim=-1)
        return r.values, r.indices

    def sort(self, x):
        return self.t.sort(x).values

    def nonzero(self, mask):
        return self.t.nonzero(mask, as_tuple=False).reshape(-1)

    def cat(self, xs, dim):
        return self.t.cat(xs, dim=dim)

    def eye4(self, n, like):
        return self.t.eye(n, device=like.device, dtype=like.dtype)[None, None]

    def eye4_new(self, n, device):
        return self.t.eye(n, device=device)[None, None]

    def zeros_long(self, n, device):
        return self.t.zeros(n, dtype=self.t.long, device=device)

    def repeat_rows(self, h, n):          # (1, H) -> (1, n, H), as EAGLE's .repeat
        return h[None].repeat(1, n, 1)

    def flatten(self, x):
        return x.reshape(-1)

    def device(self, x):
        return x.device

    def to(self, x, device):
        return x.to(device)

    def scalar(self, x):
        return float(x)

    def long_tensor(self, data, device=None):
        return self.t.tensor(data, dtype=self.t.long, device=device)

    def float_mask(self, rows):           # CPU float, like EAGLE's tree_mask
        return self.t.tensor(rows, dtype=self.t.float32)[None, None]


class NumpyOps:
    """Same operations on numpy, used by the tests (no GPU or torch needed)."""

    def __init__(self):
        import numpy as np
        self.np = np

    def topk(self, x, k):
        np = self.np
        idx = np.argsort(-x, axis=-1, kind="stable")[..., :k]
        return np.take_along_axis(x, idx, axis=-1), idx

    def sort(self, x):
        return self.np.sort(x)

    def nonzero(self, mask):
        return self.np.nonzero(mask)[0]

    def cat(self, xs, dim):
        return self.np.concatenate(xs, axis=dim)

    def eye4(self, n, like):
        return self.np.eye(n, dtype=like.dtype)[None, None]

    def eye4_new(self, n, device):
        return self.np.eye(n)[None, None]

    def zeros_long(self, n, device):
        return self.np.zeros(n, dtype=self.np.int64)

    def repeat_rows(self, h, n):
        return self.np.repeat(h[None], n, axis=1)

    def flatten(self, x):
        return x.reshape(-1)

    def device(self, x):
        return None

    def to(self, x, device):
        return x

    def scalar(self, x):
        return float(x)

    def long_tensor(self, data, device=None):
        return self.np.asarray(data, dtype=self.np.int64)

    def float_mask(self, rows):
        return self.np.asarray(rows, dtype=self.np.float32)[None, None]


# ----------------------------------------------------------------------------- selection
def select_children(ops, cu_scores, remaining: int, log_mu: float, root_k: Optional[int]):
    """One adaptive tree layer. cu_scores: (n_parents, V) cumulative log-probs of all children.

    Returns (parent_local, token_draft_idx, score), each 1-D, ordered parent-major and
    by descending score within a parent (EAGLE's node order within a layer).
    """
    n_par, vocab = cu_scores.shape
    if remaining <= 0:
        raise ValueError("no budget left")
    if root_k is not None:                       # Layer 0: robust top-K initialisation
        if n_par != 1:
            raise ValueError("the root layer has exactly one parent")
        k = min(root_k, remaining, vocab)
        vals, idx = ops.topk(cu_scores, k)       # (1, k), descending
        parent_local = ops.zeros_long(k, ops.device(cu_scores))
        return parent_local, idx[0], vals[0]

    m = min(remaining, vocab)                    # exact pre-filter (see module docstring)
    vals, idx = ops.topk(cu_scores, m)           # (n_par, m), each row descending
    anchor = vals[:, 0].max()                    # m_d
    flat_vals = ops.flatten(vals)
    passing = ops.nonzero(flat_vals >= anchor + log_mu)   # p(u) >= mu * m_d, in log space
    if passing.shape[0] > remaining:             # budget check: keep the top (N - |T|)
        _, best = ops.topk(flat_vals[passing], remaining)
        passing = passing[best]
    passing = ops.sort(passing)                  # parent-major, then rank
    parent_local = passing // m
    rank = passing % m
    return parent_local, idx[parent_local, rank], vals[parent_local, rank]


# ----------------------------------------------------------------------------- tree outputs
def build_tree_outputs(parents: List[int], depths: List[int], sort_for_sampling: bool):
    """EAGLE-format tree tensors from parent pointers (index 0 = root, parents[0] = -1).

    Returns python lists: tree_mask rows (n x n, 1 = attend), position ids (= depth)
    and retrieve indices (one root->leaf path per leaf, padded with -1), using the same
    leaf order as cnets.topK_genrate and its custom_sort when a logits processor is set.
    """
    n = len(parents)
    rows = []
    for i in range(n):
        r = [0] * n
        r[0] = 1
        j = i
        while j > 0:
            r[j] = 1
            j = parents[j]
        rows.append(r)
    has_child = set(parents[1:])
    max_depth = max(depths) + 1
    retrieve = []
    for i in range(n):
        if i in has_child:
            continue
        path = [-1] * max_depth
        j = i
        while j >= 0:
            path[depths[j]] = j
            j = parents[j]
        retrieve.append(path)
    if sort_for_sampling:
        maxitem = (n - 1) + 5                    # EAGLE: total_tokens + 5
        retrieve = sorted(retrieve, key=lambda lst: [x if x >= 0 else maxitem for x in lst])
    return rows, list(depths), retrieve


# ----------------------------------------------------------------------------- drafter
class AdaptiveDrafter:
    """Replaces ea_layer.topK_genrate with adaptive tree's Algorithm 2."""

    def __init__(self, layer, cfg: AdaptiveConfig, ops=None):
        cfg.validate()
        for attr in ("lm_head", "norm", "logsoftmax", "d2t", "reset", "config"):
            if not hasattr(layer, attr):
                raise TypeError(f"adaptive tree needs an EAGLE-3 draft layer (missing .{attr})")
        if not hasattr(layer.config, "draft_vocab_size"):
            raise TypeError("adaptive tree needs an EAGLE-3 draft layer (config.draft_vocab_size)")
        self.layer = layer
        self.cfg = cfg
        self.ops = ops or TorchOps()
        self.log_mu = math.log(cfg.mu)
        self.max_layers = cfg.max_depth or cfg.budget  # depth can never exceed N - 1 anyway
        self.last: Dict = {}

    def _fill(self, layers, room, parents, depths, token_chunks):
        """Append the `room` best unused candidates over all recorded layers as leaves."""
        ops = self.ops
        vals_all, meta = [], []
        for li, (cu, par_ids, kept, child_depth) in enumerate(layers):
            m = min(max(kept) + room, cu.shape[-1])
            vals, idx = ops.topk(cu, m)                      # (n_par, m), each row descending
            pos = _fill_candidates(kept, m)
            if not pos:
                continue
            pos_t = ops.long_tensor(pos, device=ops.device(cu))
            vals_all.append(ops.flatten(vals)[pos_t])
            meta.append((li, ops.flatten(idx)[pos_t], pos, m))
        if not vals_all:
            return 0
        flat = ops.cat(vals_all, dim=0)
        k = min(room, int(flat.shape[0]))
        _, best = ops.topk(flat, k)
        bounds, start = [], 0
        for li, toks, pos, m in meta:
            bounds.append((start, start + len(pos), li, toks, pos, m))
            start += len(pos)
        picks = []
        for b in sorted(int(x) for x in best.tolist()):
            for lo, hi, li, toks, pos, m in bounds:
                if lo <= b < hi:
                    par_ids, child_depth = layers[li][1], layers[li][3]
                    picks.append((par_ids[pos[b - lo] // m], toks[b - lo], child_depth))
                    break
        tok_t = self._to_target_ids(ops.cat([t[None] for _, t, _ in picks], dim=0))
        for parent, _, d in picks:
            parents.append(parent)
            depths.append(d)
        token_chunks.append(tok_t)
        return len(picks)

    def _to_target_ids(self, idx):
        L = self.layer
        if L.config.vocab_size == L.config.draft_vocab_size:
            return idx
        return idx + L.d2t[idx]

    def _logp(self, h):
        L = self.layer
        return L.logsoftmax(L.lm_head(L.norm(h)))

    def topK_genrate(self, hidden_states, input_ids, head, logits_processor):
        ops, L, N = self.ops, self.layer, self.cfg.budget
        dev = ops.device(hidden_states)

        # --- identical to cnets.Model.topK_genrate: context forward + root distribution
        input_ids = ops.to(input_ids, dev)
        sample_token = input_ids[:, -1]
        input_ids = input_ids[:, 1:]
        len_posi = input_ids.shape[1]
        L.reset()
        if getattr(L, "stable_kv", None) is not None:
            kv_len = L.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = L(hidden_states, input_ids=input_ids[:, kv_len:],
                                            past_key_values=L.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = L(hidden_states, input_ids=input_ids, use_cache=True)
        L.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]
        root_logp = self._logp(last_hidden)           # (1, V_draft)
        forwards = 1

        parents: List[int] = [-1]                      # index 0 = root (sample_token)
        depths: List[int] = [0]
        token_chunks = [sample_token]
        widths: List[int] = []

        # --- layer 0: robust top-K initialisation
        par, tok, score = select_children(ops, root_logp, N - 1, self.log_mu, self.cfg.root_k)
        tok_t = self._to_target_ids(tok)
        n_new = int(tok.shape[0])
        cur_global = list(range(1, 1 + n_new))
        parents += [0] * n_new
        depths += [1] * n_new
        token_chunks.append(tok_t)
        widths.append(n_new)
        cur_hidden = ops.repeat_rows(last_hidden, n_new)
        cur_ids = tok_t[None]
        cur_scores = score
        tree_mask = ops.eye4_new(n_new, dev)       # float32 eye, as EAGLE's tree_mask_init
        depth = 1
        # every layer's candidate scores, for `fill`: (cu, parent ids, children kept per parent, depth)
        layers = [(root_logp, [0], [n_new], 1)] if self.cfg.fill else None

        # --- layers d >= 1: confidence-gated expansion until |T| = N
        while len(parents) < N and depth < self.max_layers:
            L.tree_mask = tree_mask
            position_ids = len_posi + ops.zeros_long(len(cur_global), dev)
            out_hidden, past_key_values = L(cur_hidden, input_ids=cur_ids,
                                            past_key_values=past_key_values,
                                            position_ids=position_ids, use_cache=True)
            len_posi += 1
            forwards += 1
            cu = self._logp(out_hidden[0]) + cur_scores[:, None]    # (n_parents, V_draft)
            par, tok, score = select_children(ops, cu, N - len(parents), self.log_mu, None)
            tok_t = self._to_target_ids(tok)
            par_list = [int(x) for x in par.tolist()]
            n_new = len(par_list)
            if layers is not None:
                kept = [0] * len(cur_global)
                for p in par_list:
                    kept[p] += 1
                layers.append((cu, list(cur_global), kept, depth + 1))
            first = len(parents)
            parents += [cur_global[p] for p in par_list]
            depths += [depth + 1] * n_new
            token_chunks.append(tok_t)
            widths.append(n_new)
            cur_global = list(range(first, first + n_new))
            cur_hidden = out_hidden[:, par]
            cur_ids = tok_t[None]
            cur_scores = score
            tree_mask = ops.cat((tree_mask[:, :, par], ops.eye4(n_new, tree_mask)), dim=3)
            depth += 1

        # --- fill: the cap stopped the tree under budget; add the best unused candidates of the
        # layers already built as leaves (cumulative score, EAGLE-2's rule), no new draft pass
        filled = 0
        if layers is not None and len(parents) < N:
            filled = self._fill(layers, N - len(parents), parents, depths, token_chunks)

        # --- EAGLE-format outputs
        draft_tokens = ops.cat(token_chunks, dim=0)[None]
        rows, pos, retrieve = build_tree_outputs(parents, depths, logits_processor is not None)
        self.last = {"forwards": forwards, "nodes": len(parents), "depth": max(depths),
                     "widths": widths, "filled": filled}
        return (draft_tokens,
                ops.long_tensor(retrieve),
                ops.float_mask(rows),
                ops.long_tensor(pos, device=dev))


def _fill_candidates(kept, m):
    """Flat (parent * m + rank) positions of each parent's unused top-m children: TALON keeps a
    prefix of each parent's ranking (the gate and the budget cut are both monotone in score)."""
    return [p * m + r for p, c in enumerate(kept) for r in range(c, m)]


def enable_adaptive(model, cfg: AdaptiveConfig, ops=None) -> AdaptiveDrafter:
    """Patch an loaded EaModel so every draft tree is built with adaptive tree.

    Load the model with total_token = cfg.budget so EAGLE's own length guard
    (max_length - ea_layer.total_tokens - 10) matches the adaptive tree budget.
    """
    layer = model.ea_layer
    if getattr(layer, "total_tokens", cfg.budget - 1) != cfg.budget - 1:
        raise ValueError(f"load the model with total_token={cfg.budget} "
                         f"(ea_layer.total_tokens={layer.total_tokens})")
    drafter = AdaptiveDrafter(layer, cfg, ops)
    layer.topK_genrate = types.MethodType(lambda _self, *a, **k: drafter.topK_genrate(*a, **k), layer)
    layer.adaptive = drafter
    return drafter


# ----------------------------------------------------------------------------- reference
def reference_adaptive_tree(root_logp: List[float], child_logp, budget: int, mu: float, root_k: int):
    """Plain-Python Algorithm 2 over the FULL vocabulary (no pre-filter), for testing.

    root_logp: log-probs of the root's children; child_logp(path_tuple) -> list of
    log-probs for the next token given the draft path. Returns nodes as a list of
    (path_tuple, cumulative_logp) in insertion order (root excluded).
    """
    nodes = []
    V = len(root_logp)
    order = sorted(range(V), key=lambda w: (-root_logp[w], w))[:min(root_k, budget - 1, V)]
    layer = [((w,), root_logp[w]) for w in order]
    nodes += layer
    log_mu = math.log(mu)
    while 1 + len(nodes) < budget:
        cands = []
        for pi, (path, s) in enumerate(layer):
            lp = child_logp(path)
            for w in range(V):
                cands.append((pi, w, path + (w,), s + lp[w]))
        m = max(c[3] for c in cands)
        passing = [c for c in cands if c[3] >= m + log_mu]
        rem = budget - 1 - len(nodes)
        if len(passing) > rem:
            passing = sorted(passing, key=lambda c: -c[3])[:rem]
        passing = sorted(passing, key=lambda c: (c[0], -c[3], c[1]))
        layer = [(c[2], c[3]) for c in passing]
        nodes += layer
    return nodes


# ----------------------------------------------------------------------------- fixed templates
@dataclass(frozen=True)
class TemplateConfig:
    """A fixed tree shape: exactly widths[d] verified nodes at depth d + 1 (root excluded).

    branch: children each expanded node proposes (its top-`branch` draft tokens).
    expand: nodes of each layer that are drafted further, best cumulative score first
            (0 = all). EAGLE's static tree ties both to one top_k; here they are separate.
    Each layer keeps the best widths[d] of its candidates by cumulative log-prob, so the
    tree has exactly 1 + sum(widths) = budget nodes and len(widths) layers.
    """
    widths: tuple
    branch: int = 10
    expand: int = 0
    label: str = ""

    @property
    def budget(self) -> int:
        return 1 + sum(self.widths)

    @property
    def root_k(self) -> int:              # sizes EAGLE's unused static buffers at load time
        return self.widths[0]

    def validate(self) -> "TemplateConfig":
        if not self.widths or any(int(w) < 1 for w in self.widths):
            raise ValueError("widths must be positive")
        if self.branch < 1 or self.expand < 0:
            raise ValueError("branch must be >= 1 and expand >= 0")
        for d in range(1, len(self.widths)):
            n_exp = self.widths[d - 1] if not self.expand else min(self.expand, self.widths[d - 1])
            if self.widths[d] > n_exp * self.branch:
                raise ValueError(f"layer {d + 1}: width {self.widths[d]} > {n_exp} expanded x "
                                 f"{self.branch} children")
        return self

    @property
    def name(self) -> str:
        tag = self.label or "w" + "-".join(str(w) for w in self.widths)
        return f"tmpl_{tag}_b{self.branch}_e{self.expand or 'all'}"

    def to_dict(self) -> Dict:
        return {"widths": list(self.widths), "branch": self.branch, "expand": self.expand,
                "budget": self.budget, "label": self.label, "name": self.name}


PROFILES = {
    "down": lambda d, L: L - d + 1,               # wide at the top, narrowing
    "flat": lambda d, L: 1,
    "up": lambda d, L: d,                         # narrow at the top, widening (EAGLE-3-like)
    "diamond": lambda d, L: min(d, L - d + 1),    # widest in the middle
}


def profile_widths(profile: str, depth: int, budget: int) -> tuple:
    """`depth` layer widths following `profile`, scaled to hold exactly budget - 1 nodes
    (largest-remainder rounding, every layer at least 1)."""
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}")
    n = budget - 1
    if not 1 <= depth <= n:
        raise ValueError(f"depth must be in 1..{n}")
    f = [PROFILES[profile](d, depth) for d in range(1, depth + 1)]
    raw = [n * x / sum(f) for x in f]
    w = [max(1, int(r)) for r in raw]
    order = sorted(range(depth), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    i = 0
    while sum(w) < n:
        w[order[i % depth]] += 1
        i += 1
    while sum(w) > n:                              # only when the max(1, .) floor overshoots
        j = max(range(depth), key=lambda k: w[k])
        w[j] -= 1
    if profile in ("down", "up"):                  # rounding must not break the monotone shape
        w = sorted(w, reverse=(profile == "down"))
    return tuple(w)


def geometric_widths(first: int, ratio: float, budget: int) -> tuple:
    """Widths first, first*ratio, first*ratio^2, ... (rounded, at least 1) until they hold
    budget - 1 nodes; the last layer is trimmed so the total is exact."""
    widths, left, w = [], budget - 1, float(first)
    while left > 0:
        n = min(max(1, int(round(w))), left)
        widths.append(n)
        left -= n
        w *= ratio
        if len(widths) > budget:
            raise ValueError("ratio too small to fill the budget")
    return tuple(widths)


class TemplateDrafter(AdaptiveDrafter):
    """Builds every draft tree from a fixed TemplateConfig (same EAGLE plumbing as adaptive)."""

    def __init__(self, layer, cfg: TemplateConfig, ops=None):
        super().__init__(layer, AdaptiveConfig(cfg.budget, 1.0, cfg.widths[0]), ops)
        self.tmpl = cfg

    def _expanded(self, score, n):
        """Local indices of the nodes to draft further: the best `expand` by score, in order."""
        e = self.tmpl.expand
        if not e or e >= n:
            return None                                  # all of them
        _, best = self.ops.topk(score, e)
        return self.ops.sort(best)

    def topK_genrate(self, hidden_states, input_ids, head, logits_processor):
        ops, L, T = self.ops, self.layer, self.tmpl
        dev = ops.device(hidden_states)

        # --- identical to cnets.Model.topK_genrate: context forward + root distribution
        input_ids = ops.to(input_ids, dev)
        sample_token = input_ids[:, -1]
        input_ids = input_ids[:, 1:]
        len_posi = input_ids.shape[1]
        L.reset()
        if getattr(L, "stable_kv", None) is not None:
            kv_len = L.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = L(hidden_states, input_ids=input_ids[:, kv_len:],
                                            past_key_values=L.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = L(hidden_states, input_ids=input_ids, use_cache=True)
        L.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]
        root_logp = self._logp(last_hidden)
        forwards = 1

        parents: List[int] = [-1]
        depths: List[int] = [0]
        token_chunks = [sample_token]

        # --- layer 1: the root's best widths[0] children
        vals, idx = ops.topk(root_logp, T.widths[0])
        tok, score = idx[0], vals[0]
        n_new = int(tok.shape[0])
        first = len(parents)
        parents += [0] * n_new
        depths += [1] * n_new
        token_chunks.append(self._to_target_ids(tok))
        ex = self._expanded(score, n_new)
        sel = ex if ex is not None else ops.long_tensor(list(range(n_new)), device=dev)
        cur_global = [first + int(i) for i in sel.tolist()]
        cur_ids = self._to_target_ids(tok[sel])[None]
        cur_scores = score[sel]
        cur_hidden = ops.repeat_rows(last_hidden, len(cur_global))
        tree_mask = ops.eye4_new(len(cur_global), dev)

        # --- layers 2..: best widths[d] of the expanded nodes' top-`branch` children
        for d in range(1, len(T.widths)):
            L.tree_mask = tree_mask
            position_ids = len_posi + ops.zeros_long(len(cur_global), dev)
            out_hidden, past_key_values = L(cur_hidden, input_ids=cur_ids,
                                            past_key_values=past_key_values,
                                            position_ids=position_ids, use_cache=True)
            len_posi += 1
            forwards += 1
            cu = self._logp(out_hidden[0]) + cur_scores[:, None]          # (n_expanded, V)
            b = min(T.branch, cu.shape[-1])
            vals, idx = ops.topk(cu, b)                                    # (n_expanded, b)
            _, pos = ops.topk(ops.flatten(vals), T.widths[d])
            pos = ops.sort(pos)                                            # parent-major
            par, rank = pos // b, pos % b
            tok, score = idx[par, rank], vals[par, rank]
            n_new = int(tok.shape[0])
            first = len(parents)
            parents += [cur_global[int(p)] for p in par.tolist()]
            depths += [d + 1] * n_new
            tok_t = self._to_target_ids(tok)
            token_chunks.append(tok_t)
            if d == len(T.widths) - 1:
                break
            ex = self._expanded(score, n_new)
            sel = ex if ex is not None else ops.long_tensor(list(range(n_new)), device=dev)
            cur_global = [first + int(i) for i in sel.tolist()]
            cur_hidden = out_hidden[:, par[sel]]
            cur_ids = tok_t[sel][None]
            cur_scores = score[sel]
            tree_mask = ops.cat((tree_mask[:, :, par[sel]], ops.eye4(len(cur_global), tree_mask)), dim=3)

        draft_tokens = ops.cat(token_chunks, dim=0)[None]
        rows, pos_ids, retrieve = build_tree_outputs(parents, depths, logits_processor is not None)
        self.last = {"forwards": forwards, "nodes": len(parents), "depth": max(depths),
                     "widths": list(T.widths), "parents": parents, "depths": depths}
        return (draft_tokens,
                ops.long_tensor(retrieve),
                ops.float_mask(rows),
                ops.long_tensor(pos_ids, device=dev))


def enable_template(model, cfg: TemplateConfig, ops=None) -> TemplateDrafter:
    """Patch a loaded EaModel so every draft tree follows `cfg` (load with total_token = cfg.budget)."""
    layer = model.ea_layer
    if getattr(layer, "total_tokens", cfg.budget - 1) != cfg.budget - 1:
        raise ValueError(f"load the model with total_token={cfg.budget} "
                         f"(ea_layer.total_tokens={layer.total_tokens})")
    drafter = TemplateDrafter(layer, cfg, ops)
    layer.topK_genrate = types.MethodType(lambda _self, *a, **k: drafter.topK_genrate(*a, **k), layer)
    layer.adaptive = drafter          # instrument_drafting reads .last from here
    return drafter
