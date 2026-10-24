"""TALON-S (specdec/sglang_talon_plugin.py) against a CPU copy of SGLang v0.5.22's tree builder.

`_first`, `_later` and `_organize` reproduce speculative/spec_utils.py `_select_top_k_tokens_first`,
`_select_top_k_tokens_later` and eagle_utils.py `organize_draft_results` (hidden states dropped,
fast_topk -> torch.topk). Random draft distributions drive both the plain EAGLE-2 rule and the
hooked one; the tests check the verified set is a valid tree and follows TALON's rules.
"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from specdec.sglang_talon_plugin import TalonSConfig, make_hook, talon_rescore  # noqa: E402


def _first(topk_p, topk_index, topk):
    tree_info = (topk_p.unsqueeze(1), topk_index,
                 torch.arange(-1, topk, dtype=torch.long).expand(topk_p.shape[0], -1))
    return topk_index.flatten(), None, topk_p, tree_info


def _later(i, topk_p, topk_index, scores, topk):
    topk_sq = topk * topk
    expand_scores = scores.unsqueeze(2) * topk_p.view(-1, topk, topk)
    topk_cs_p, topk_cs_index = torch.topk(expand_scores.flatten(start_dim=1), topk, dim=-1)
    topk_index = topk_index.view(-1, topk_sq)
    input_ids = torch.gather(topk_index, 1, topk_cs_index).flatten()
    tree_info = (expand_scores, topk_index, topk_cs_index + (topk_sq * (i - 1) + topk))
    return input_ids, None, topk_cs_p, tree_info


def select_top_k_tokens(i, topk_p, topk_index, hidden_states, scores, topk):
    if i == 0:
        return _first(topk_p, topk_index, topk)
    return _later(i, topk_p, topk_index, scores, topk)


def _organize(score_list, token_list, parents_list, num_draft_token):
    score_list = torch.cat(score_list, dim=1).flatten(1)
    top = torch.sort(torch.topk(score_list, num_draft_token - 1, dim=-1).indices).values
    return top, torch.cat(parents_list[:-1], dim=1)


def draft(select, b, topk, steps, budget, seed, peaked=False):
    """Run the drafting loop on random distributions; returns verified candidate ids,
    parents_list, raw cumulative probabilities and the per-layer keys."""
    g = torch.Generator().manual_seed(seed)
    vocab = 50

    def dist(n):
        logits = torch.randn(n, vocab, generator=g) * (6.0 if peaked else 2.0)
        return torch.topk(torch.softmax(logits, -1), topk, dim=-1)

    topk_p, topk_index = dist(b)
    scores, score_list, token_list, parents_list, raw = None, [], [], [], []
    for i in range(steps):
        raw_scores = (topk_p.unsqueeze(1) if i == 0 else
                      scores.unsqueeze(2) * topk_p.view(-1, topk, topk))
        raw.append(raw_scores.abs().flatten(1))
        _, _, scores, info = select(i, topk_p, topk_index, None, scores, topk)
        score_list.append(info[0]), token_list.append(info[1]), parents_list.append(info[2])
        if i == steps - 1:
            break
        topk_p, topk_index = dist(b * topk)
    sel, parents = _organize(score_list, token_list, parents_list, budget)
    return sel, parents_list, torch.cat(raw, 1), score_list


def parent_of(c, parents_list, topk):
    """Candidate id of c's parent (-1 = root), from SGLang's parents_list layout."""
    if c < topk:
        return -1
    layer = 1 + (c - topk) // (topk * topk)
    j = (c - topk) % (topk * topk)
    beam = j // topk
    return int(parents_list[layer - 1][beam + (1 if layer == 1 else 0)])


def layer_of(c, topk):
    return 0 if c < topk else 1 + (c - topk) // (topk * topk)


@unittest.skipIf(torch is None, "needs torch")
class TestTalonS(unittest.TestCase):
    B, K, STEPS, N, MU = 4, 10, 6, 40, 0.03

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()  # markers from make_hook land here, not in the repo
        self.addCleanup(tmp.cleanup)
        os.environ["SPECDEC_TALON_OUT_DIR"] = tmp.name

    def hooked(self, mu=None):
        hook = make_hook(TalonSConfig(mu=self.MU if mu is None else mu, num_steps=self.STEPS,
                                      root_k=self.K))
        return lambda *a: hook(select_top_k_tokens, *a)

    def test_tree_valid_and_rules(self):
        for seed in range(20):
            for peaked in (False, True):
                sel, parents_list, raw, keys = draft(self.hooked(), self.B, self.K, self.STEPS,
                                                     self.N, seed, peaked)
                keys = torch.cat(keys, 1).flatten(1)
                for r in range(self.B):
                    chosen = set(sel[r].tolist())
                    # 1) valid tree: every verified node's parent is verified (or the root)
                    for c in chosen:
                        p = parent_of(c, [x[r] for x in parents_list], self.K)
                        self.assertTrue(p == -1 or p in chosen, (seed, r, c, p))
                    # 2) robust init: all K root children verified
                    self.assertTrue(set(range(self.K)) <= chosen)
                    # 3) layer-first budget: no kept node is skipped while a deeper or gated
                    #    node is verified
                    kept = (keys[r] >= 1.0)
                    for c in chosen:
                        if not kept[c]:  # a filler: then every kept node must be verified
                            self.assertTrue(set(torch.nonzero(kept).flatten().tolist()) <= chosen)
                    kept_unchosen = [c for c in torch.nonzero(kept).flatten().tolist() if c not in chosen]
                    if kept_unchosen:
                        shallowest = min(layer_of(c, self.K) for c in kept_unchosen)
                        deeper = [c for c in chosen if layer_of(c, self.K) > shallowest]
                        self.assertEqual(deeper, [], (seed, r))
                    # 4) gate: every kept node passes p >= mu * (best kept-parent child of its layer)
                    for c in torch.nonzero(kept).flatten().tolist():
                        d = layer_of(c, self.K)
                        if d == 0:
                            continue
                        lo = self.K + (d - 1) * self.K * self.K
                        layer_keys = keys[r, lo:lo + self.K * self.K]
                        self.assertGreaterEqual(float(layer_keys[c - lo]) - (self.STEPS - d), self.MU - 1e-6)
                        # 5) only kept nodes are expanded: a kept node's parent is kept
                        p = parent_of(c, [x[r] for x in parents_list], self.K)
                        self.assertTrue(p == -1 or bool(kept[p]), (seed, r, c, p))

    def test_changes_the_verified_set_vs_eagle2(self):
        differs = 0
        for seed in range(10):
            sel_t, *_ = draft(self.hooked(), self.B, self.K, self.STEPS, self.N, seed)
            sel_e, *_ = draft(select_top_k_tokens, self.B, self.K, self.STEPS, self.N, seed)
            differs += int(not torch.equal(sel_t, sel_e))
        self.assertGreater(differs, 0)  # the hook changes which nodes get verified

    def test_graph_safe_shapes_and_dtypes(self):
        s0 = torch.rand(3, 1, self.K, dtype=torch.float16)
        k0, b0 = talon_rescore(0, s0, s0[:, 0], self.STEPS, self.MU)
        self.assertEqual((k0.shape, k0.dtype, b0.dtype), (s0.shape, torch.float32, torch.float16))
        s1 = torch.rand(3, self.K, self.K)
        k1, b1 = talon_rescore(2, s1, torch.rand(3, self.K), self.STEPS, self.MU)
        self.assertEqual((k1.shape, b1.shape), (s1.shape, (3, self.K)))

    def test_rejects_topk_mismatch(self):
        hook = make_hook(TalonSConfig(mu=0.03, num_steps=3, root_k=10))
        p, idx = torch.topk(torch.rand(2, 50), 8, dim=-1)
        with self.assertRaises(ValueError):
            hook(select_top_k_tokens, 0, p, idx, None, None, 8)

    def test_disabled_without_env(self):
        os.environ.pop("SPECDEC_TALON", None)
        self.assertIsNone(TalonSConfig.from_env())


if __name__ == "__main__":
    unittest.main()
