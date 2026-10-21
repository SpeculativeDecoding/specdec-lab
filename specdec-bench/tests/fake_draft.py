"""A fake EAGLE-3 draft layer (cnets.Model interface) for testing adaptive tree without a GPU.

It reproduces the interface adaptive tree uses: __call__(hidden_states, input_ids,
past_key_values, position_ids, use_cache) -> (out_hidden, past), lm_head, norm,
logsoftmax, d2t, config.{vocab_size, draft_vocab_size}, reset(), stable_kv, tree_mask.

Crucially, it does NOT trust the caller about tree structure. For every tree node it
reconstructs the node's token path from the attention mask it is given and the KV
entries it holds, then checks that:
  * the node attends to itself and only to tree entries that form a chain of ancestors,
  * the hidden state passed in for the node is the output hidden of its parent path,
  * all nodes of one call share a position id, one more than the previous layer,
  * the persistent stable_kv only ever contains context entries.
The next-token distribution it returns is a function of that reconstructed path, so a
wrong mask, hidden routing or token map changes the tree and the tests catch it.
Works with numpy arrays, and with torch tensors when torch is installed.
"""
import hashlib

import numpy as np


def _np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class _Cfg:
    def __init__(self, vocab_size, draft_vocab_size):
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size


def path_logits(path, vocab, seed=0, peaky=None):
    """Deterministic logits for a draft-id path. `peaky(path)` -> scale (bigger = sharper)."""
    h = int(hashlib.sha256(f"{seed}:{path}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    scale = peaky(path) if peaky else rng.choice([0.5, 2.0, 6.0])
    return rng.normal(size=vocab) * scale


class FakeDraftLayer:
    def __init__(self, draft_vocab, use_d2t=False, seed=0, peaky=None, backend="numpy",
                 logits_fn=None):
        self.V = draft_vocab
        self.seed = seed
        self.peaky = peaky
        self.logits_fn = logits_fn
        self.backend = backend
        if use_d2t:
            # draft id i <-> target id 3*i + 5  (d2t stores target - draft)
            self.target_of = np.arange(draft_vocab) * 3 + 5
            vocab_size = int(self.target_of[-1]) + 10
        else:
            self.target_of = np.arange(draft_vocab)
            vocab_size = draft_vocab
        self.draft_of = {int(t): i for i, t in enumerate(self.target_of)}
        self.config = _Cfg(vocab_size, draft_vocab)
        self.d2t = self._out(self.target_of - np.arange(draft_vocab))
        self.tree_mask = None
        self.stable_kv = None
        self.paths = [()]            # registry: hidden "vector" = id of a draft path
        self.calls = 0
        self.tree_calls = 0
        self._pos_prev = None

    # ---- backend helpers
    def _out(self, a):
        if self.backend == "torch":
            import torch
            return torch.as_tensor(a)
        return a

    def _reg(self, path):
        self.paths.append(tuple(path))
        return len(self.paths) - 1

    # ---- interface used by adaptive tree
    def reset(self):
        self.tree_mask = None

    def reset_kv(self):                     # eagenerate calls this before every generation
        self.stable_kv = None

    def norm(self, h):
        return h

    def lm_head(self, h):
        ids = _np(h).reshape(-1).astype(int)
        rows = []
        for i in ids:
            p = self.paths[i]
            rows.append(self.logits_fn(p) if self.logits_fn else
                        path_logits(p, self.V, self.seed, self.peaky))
        return self._out(np.stack(rows))

    def logsoftmax(self, x):
        a = _np(x).astype(np.float64)
        a = a - a.max(-1, keepdims=True)
        return self._out(a - np.log(np.exp(a).sum(-1, keepdims=True)))

    def logp_of_path(self, path):
        """log P(path[-1] | path[:-1]) under this draft model (for tests)."""
        lg = self.logits_fn(path[:-1]) if self.logits_fn else path_logits(path[:-1], self.V, self.seed, self.peaky)
        lg = lg - lg.max()
        return float(lg[path[-1]] - np.log(np.exp(lg).sum()))

    def __call__(self, hidden_states, input_ids, past_key_values=None, position_ids=None,
                 use_cache=True):
        self.calls += 1
        ids = _np(input_ids).reshape(-1).tolist()
        hid = _np(hidden_states)
        old = list(past_key_values[0][1]) if past_key_values is not None else []
        if self.tree_mask is None:                                   # context forward
            assert all(e["kind"] == "ctx" for e in old), "stable_kv polluted by tree entries"
            assert hid.shape[1] == len(ids), "context hidden/input length mismatch"
            new = old + [{"kind": "ctx", "token": t} for t in ids]
            out = np.zeros((1, len(ids), 1))
            out[0, -1, 0] = 0                                        # root path ()
            self._pos_prev = None
            return self._out(out), self._pack(new)

        self.tree_calls += 1
        tm = _np(self.tree_mask)
        n = len(ids)
        assert tm.shape[2] == n, "tree mask rows != number of queried nodes"
        new = old + [{"kind": "tree", "token": t} for t in ids]
        cols = tm.shape[3]
        window = new[len(new) - cols:]
        assert all(e["kind"] == "tree" for e in window), "tree mask columns cover context"
        pos = _np(position_ids).reshape(-1).tolist()
        assert len(set(pos)) == 1, "nodes of one layer must share a position"
        if self._pos_prev is not None:
            assert pos[0] == self._pos_prev + 1, "position must advance by one per layer"
        self._pos_prev = pos[0]
        out = np.zeros((1, n, 1))
        for j in range(n):
            att = [c for c in range(cols) if tm[0, 0, j, c] == 1]
            assert att and att[-1] == cols - n + j, "node must attend to itself last"
            path_t = [window[c]["token"] for c in att]
            path = tuple(self.draft_of[int(t)] for t in path_t)
            parent_path = self.paths[int(hid[0, j, 0])]
            assert parent_path == path[:-1], f"hidden routed from {parent_path}, mask says {path[:-1]}"
            out[0, j, 0] = self._reg(path)
        return self._out(out), self._pack(new)

    def _pack(self, entries):
        k = np.zeros((1, 1, len(entries), 1))
        return ((self._out(k), entries),)
