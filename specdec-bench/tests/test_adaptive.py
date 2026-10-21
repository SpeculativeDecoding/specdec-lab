"""adaptive tree tests. Run with numpy always, and additionally with torch when it is installed:

  python -m unittest tests.test_adaptive -v
"""
import itertools
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from fake_draft import FakeDraftLayer, _np  # noqa: E402
from specdec.adaptive import (NumpyOps, AdaptiveConfig, AdaptiveDrafter, build_tree_outputs,  # noqa: E402
                           enable_adaptive, reference_adaptive_tree)

try:
    import torch  # noqa: F401
    BACKENDS = ["numpy", "torch"]
except Exception:
    BACKENDS = ["numpy"]


def make_ops(backend):
    if backend == "torch":
        from specdec.adaptive import TorchOps
        return TorchOps()
    return NumpyOps()


def as_backend(a, backend):
    if backend == "torch":
        import torch
        return torch.as_tensor(a)
    return a


def run_adaptive(layer, cfg, backend, prompt_len=6, logits_processor=None, input_ids=None):
    d = AdaptiveDrafter(layer, cfg, make_ops(backend))
    if input_ids is None:
        input_ids = np.array([[int(layer.target_of[i % layer.V]) for i in range(prompt_len + 1)]])
    hidden = np.zeros((1, input_ids.shape[1] - 1, 1))
    out = d.topK_genrate(as_backend(hidden, backend), as_backend(input_ids, backend), None, logits_processor)
    return d, [ _np(x) for x in out ], input_ids


def tree_paths(layer, draft_tokens, parents):
    """Draft-id path of every non-root node, from the returned tokens + parent pointers."""
    toks = [layer.draft_of[int(t)] for t in draft_tokens[0][1:]]
    paths = []
    for i in range(1, len(parents)):
        p, j = [], i
        while j > 0:
            p.append(toks[j - 1])
            j = parents[j]
        paths.append(tuple(reversed(p)))
    return paths


def parents_from_mask(mask):
    """Recover parent pointers from the returned n x n mask (ancestor-or-self rows)."""
    m = mask[0, 0]
    n = m.shape[0]
    par = [-1]
    for i in range(1, n):
        anc = [j for j in range(i) if m[i, j] == 1]
        par.append(max(anc))          # nearest ancestor = deepest one = largest index
    return par


# A literal port of the leaf / retrieve-index code in EAGLE cnets.Model.topK_genrate,
# applied to EAGLE's own intermediate (mask_index_list = parent position of node i+1).
def eagle_retrieve_port(mask_index_list, sort_for_sampling):
    total_tokens = len(mask_index_list)
    tree_mask = [[False] * (total_tokens + 1) for _ in range(total_tokens + 1)]
    for i in range(total_tokens + 1):
        tree_mask[i][i] = True
        tree_mask[i][0] = True
    for i in range(total_tokens):
        tree_mask[i + 1] = [a or b for a, b in zip(tree_mask[i + 1], tree_mask[mask_index_list[i]])]
    tree_position_ids = [sum(r) - 1 for r in tree_mask]
    max_depth = max(tree_position_ids) + 1
    noleaf_index = sorted(set(mask_index_list))
    noleaf_num = len(noleaf_index) - 1
    leaf_num = total_tokens - noleaf_num
    retrieve_indices = [[-1] * max_depth for _ in range(leaf_num)]
    rid = 0
    for i in range(total_tokens + 1):
        if i not in noleaf_index:
            cid = i
            depth = tree_position_ids[i]
            for j in reversed(range(depth + 1)):
                retrieve_indices[rid][j] = cid
                cid = mask_index_list[cid - 1]
            rid += 1
    if sort_for_sampling:
        maxitem = total_tokens + 5
        retrieve_indices = sorted(retrieve_indices, key=lambda lst: [x if x >= 0 else maxitem for x in lst])
    return [[int(x) for x in r] for r in tree_mask], tree_position_ids, retrieve_indices


class TestAdaptiveAgainstReference(unittest.TestCase):
    def check(self, backend, V, cfg, use_d2t, seed, peaky=None):
        layer = FakeDraftLayer(V, use_d2t=use_d2t, seed=seed, peaky=peaky, backend=backend)
        d, (draft_tokens, retrieve, mask, pos), input_ids = run_adaptive(layer, cfg, backend)
        ctx = f"backend={backend} V={V} {cfg} d2t={use_d2t} seed={seed}"
        # 1) exact budget: N nodes including the root
        self.assertEqual(draft_tokens.shape, (1, cfg.budget), ctx)
        self.assertEqual(int(draft_tokens[0, 0]), int(input_ids[0, -1]), ctx)  # root = sample token
        # 2) same tree as the full-vocabulary reference of Algorithm 2, in the same order
        parents = parents_from_mask(mask)
        got = tree_paths(layer, draft_tokens, parents)
        root_lp = [layer.logp_of_path((w,)) for w in range(V)]
        ref = reference_adaptive_tree(root_lp, lambda p: [layer.logp_of_path(p + (w,)) for w in range(V)],
                                   cfg.budget, cfg.mu, cfg.root_k)
        self.assertEqual(got, [p for p, _ in ref], ctx)
        # 3) outputs agree with EAGLE's own post-processing given the same parents
        ermask, epos, eret = eagle_retrieve_port(parents[1:], sort_for_sampling=False)
        self.assertEqual(mask[0, 0].astype(int).tolist(), ermask, ctx)
        self.assertEqual(pos.tolist(), epos, ctx)
        self.assertEqual(retrieve.tolist(), eret, ctx)
        # 4) bookkeeping: one draft forward for the context + one per expanded layer
        self.assertEqual(d.last["forwards"], 1 + (len(d.last["widths"]) - 1), ctx)
        self.assertEqual(layer.tree_calls, len(d.last["widths"]) - 1, ctx)
        self.assertEqual(sum(d.last["widths"]), cfg.budget - 1, ctx)
        self.assertEqual(d.last["depth"], max(pos.tolist()), ctx)
        return d

    def test_grid(self):
        n = 0
        for backend in BACKENDS:
            for V, N, mu, K, use_d2t, seed in itertools.product(
                    (6, 40), (2, 5, 16, 60), (0.01, 0.03, 0.3, 1.0), (1, 3, 10), (False, True), (0, 1)):
                self.check(backend, V, AdaptiveConfig(N, mu, K), use_d2t, seed)
                n += 1
        self.assertGreater(n, 300)

    def test_paper_default(self):
        for backend in BACKENDS:
            for seed in range(10):
                self.check(backend, 300, AdaptiveConfig(60, 0.03, 10), True, seed)

    def test_prefilter_exact_with_flat_children(self):
        # One dominant parent whose children are almost uniform over a large vocabulary:
        # far more than M=remaining children pass the gate, so only the pre-filter's
        # exactness argument keeps the result identical to the full-vocab reference.
        def peaky(path):
            if len(path) == 0:
                return 8.0           # confident root
            if len(path) == 1:
                return 0.01          # nearly flat children
            return 3.0
        for backend in BACKENDS:
            self.check(backend, 500, AdaptiveConfig(60, 0.03, 10), False, 7, peaky=peaky)
            self.check(backend, 500, AdaptiveConfig(60, 0.001, 3), False, 8, peaky=peaky)


class TestAdaptiveBehaviour(unittest.TestCase):
    def test_limits(self):
        for backend in BACKENDS:
            # mu = 1 keeps only the anchor: K-wide root, then a single chain
            layer = FakeDraftLayer(50, seed=3, backend=backend)
            d, (tok, *_), _ = run_adaptive(layer, AdaptiveConfig(30, 1.0, 5), backend)
            self.assertEqual(d.last["widths"], [5] + [1] * 24)
            self.assertEqual(d.last["depth"], 25)
            # mu -> 0 admits everything: the budget fills at layer 2 (shallow and wide)
            layer = FakeDraftLayer(50, seed=3, backend=backend)
            d, _, _ = run_adaptive(layer, AdaptiveConfig(30, 1e-12, 5), backend)
            self.assertEqual(d.last["widths"], [5, 24])

    def test_root_is_robust_to_overconfidence(self):
        def logits_fn(path):
            v = np.full(20, -10.0)
            v[(len(path) * 7) % 20] = 10.0          # > 0.99 on one token everywhere
            return v
        for backend in BACKENDS:
            layer = FakeDraftLayer(20, logits_fn=logits_fn, backend=backend)
            d, _, _ = run_adaptive(layer, AdaptiveConfig(16, 0.03, 10), backend)
            self.assertEqual(d.last["widths"][0], 10)        # forced top-K at the root
            self.assertTrue(all(w == 1 for w in d.last["widths"][1:]))  # then deep and narrow

    def test_small_budget_never_forwards_tree(self):
        for backend in BACKENDS:
            layer = FakeDraftLayer(30, seed=1, backend=backend)
            d, (tok, *_), _ = run_adaptive(layer, AdaptiveConfig(6, 0.03, 10), backend)
            self.assertEqual(d.last["widths"], [5])          # top-K capped by N - 1
            self.assertEqual(layer.tree_calls, 0)

    def test_max_depth_cap(self):
        layer = FakeDraftLayer(30, seed=2)
        d, (tok, *_), _ = run_adaptive(layer, AdaptiveConfig(40, 1.0, 2, max_depth=4), "numpy")
        self.assertEqual(d.last["depth"], 4)
        self.assertEqual(tok.shape[1], 1 + 2 + 1 + 1 + 1)

    def test_sampling_sort_matches_eagle(self):
        layer = FakeDraftLayer(40, seed=5)
        d, (tok, retrieve, mask, pos), _ = run_adaptive(layer, AdaptiveConfig(32, 0.05, 6), "numpy",
                                                     logits_processor=object())
        parents = parents_from_mask(mask)
        _, _, eret = eagle_retrieve_port(parents[1:], sort_for_sampling=True)
        self.assertEqual(retrieve.tolist(), eret)

    def test_successive_calls_keep_stable_kv_clean(self):
        # Mimic eagenerate: initial call, then a call after 3 accepted tokens + new sample.
        layer = FakeDraftLayer(40, seed=9)
        d = AdaptiveDrafter(layer, AdaptiveConfig(20, 0.03, 4), NumpyOps())
        ids = np.array([[1, 2, 3, 4, 5, 6]])
        d.topK_genrate(np.zeros((1, 5, 1)), ids, None, None)
        ids2 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        d.topK_genrate(np.zeros((1, 3, 1)), ids2, None, None)   # kv_len 5 -> 3 new positions
        self.assertEqual(len(layer.stable_kv[0][1]), 8)
        self.assertTrue(all(e["kind"] == "ctx" for e in layer.stable_kv[0][1]))

    def test_config_validation(self):
        for bad in (AdaptiveConfig(1), AdaptiveConfig(60, 0.0), AdaptiveConfig(60, 1.5), AdaptiveConfig(60, 0.03, 0)):
            with self.assertRaises(ValueError):
                bad.validate()
        self.assertEqual(AdaptiveConfig().name, "adaptive_n60_mu0.03_k10")


class TestEnableAdaptive(unittest.TestCase):
    def test_patch_and_keyword_call(self):
        layer = FakeDraftLayer(40, seed=4)
        layer.total_tokens = 59                      # what EaModel sets for total_token=60

        class M:
            pass
        m = M()
        m.ea_layer = layer
        drafter = enable_adaptive(m, AdaptiveConfig(60, 0.03, 10), NumpyOps())
        self.assertIs(layer.adaptive, drafter)
        ids = np.array([[1, 2, 3, 4]])
        # update_inference_inputs calls it with keyword arguments
        out = layer.topK_genrate(np.zeros((1, 3, 1)), input_ids=ids, head=None, logits_processor=None)
        self.assertEqual(out[0].shape, (1, 60))
        layer.total_tokens = 30
        with self.assertRaises(ValueError):
            enable_adaptive(m, AdaptiveConfig(60), NumpyOps())

    def test_rejects_eagle1_layer(self):
        class NotEagle3:
            pass
        with self.assertRaises(TypeError):
            AdaptiveDrafter(NotEagle3(), AdaptiveConfig(), NumpyOps())


class TestBuildTreeOutputs(unittest.TestCase):
    def test_against_eagle_port_random_trees(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            n = int(rng.integers(2, 40))
            depths, parents = [0], [-1]
            for i in range(1, n):
                p = int(rng.integers(0, i))
                parents.append(p)
                depths.append(depths[p] + 1)
            # adaptive tree appends layer by layer; reorder nodes breadth-first like the drafter
            order = sorted(range(n), key=lambda i: (depths[i], i))
            new_of = {o: k for k, o in enumerate(order)}
            par2 = [-1] + [new_of[parents[o]] for o in order[1:]]
            dep2 = [depths[o] for o in order]
            for sort in (False, True):
                rows, pos, ret = build_tree_outputs(par2, dep2, sort)
                erows, epos, eret = eagle_retrieve_port(par2[1:], sort)
                self.assertEqual(rows, erows)
                self.assertEqual(pos, epos)
                self.assertEqual(ret, eret)




EAGLE_REPO = os.environ.get("EAGLE_REPO", os.path.expanduser("~/EAGLE"))


@unittest.skipUnless(os.path.isdir(os.path.join(EAGLE_REPO, "eagle", "model")), "EAGLE checkout not found")
class TestAdaptiveMirrorsEagleSource(unittest.TestCase):
    """adaptive tree copies these operations from cnets.Model.topK_genrate; fail if EAGLE changes them."""

    LINES = [
        "input_ids = input_ids.to(hidden_states.device)",
        "sample_token = input_ids[:, -1]",
        "input_ids = input_ids[:, 1:]",
        "len_posi = input_ids.shape[1]",
        "kv_len = self.stable_kv[0][0].shape[2]",
        "self.stable_kv = past_key_values",
        "last_hidden = out_hidden[:, -1]",
        "last_headout = self.lm_head(self.norm(last_hidden))",
        "position_ids = len_posi + self.position_ids",
        "len_posi += 1",
        "input_hidden = last_hidden[None].repeat(1, top_k, 1)",
        "input_hidden = out_hidden[:, out_ids]",
        "tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)",
        "last_headout = self.lm_head(self.norm(out_hidden[0]))",
        "cu_scores = topk_p + scores[:, None]",
        "input_ids = topk_index+self.d2t[topk_index]",
        "if self.config.vocab_size==self.config.draft_vocab_size:",
        "self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]",
        "tree_mask[:, 0] = True",
        "tree_position_ids = torch.sum(tree_mask, dim=1) - 1",
        "maxitem = total_tokens + 5",
        "retrieve_indices = sorted(retrieve_indices, key=custom_sort)",
        "return draft_tokens, retrieve_indices, tree_mask, tree_position_ids",
    ]

    def test_lines_present(self):
        src = open(os.path.join(EAGLE_REPO, "eagle", "model", "cnets.py")).read()
        for line in self.LINES:
            self.assertIn(line, src)

    def test_call_sites(self):
        u = open(os.path.join(EAGLE_REPO, "eagle", "model", "utils.py")).read()
        self.assertIn("model.ea_layer.topK_genrate(hidden_states, input_ids, model.base_model.lm_head,logits_processor)", u)
        self.assertIn("input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),", u)
        self.assertIn("head=model.base_model.lm_head,logits_processor=logits_processor)", u)
        e = open(os.path.join(EAGLE_REPO, "eagle", "model", "ea_model.py")).read()
        self.assertIn("max_length = max_length - self.ea_layer.total_tokens - 10", e)




class _FakeEaModelWithLayer:
    """eagenerate drives ea_layer.topK_genrate like EAGLE: once after prefill, once per step."""

    def __init__(self, layer, steps=4):
        from fakes import FakeTokenizer
        self.ea_layer = layer
        self.tok = FakeTokenizer()
        self.steps = steps

    def get_tokenizer(self):
        return self.tok

    def eagenerate(self, input_ids, temperature=0.0, top_p=0.0, top_k=0.0,
                   max_new_tokens=512, max_length=2048, log=False, is_llama3=False):
        ids = np.asarray(input_ids)
        self.ea_layer.reset_kv()                  # as EaModel.eagenerate does
        hidden = np.zeros((1, ids.shape[1], 1))   # target hidden for every prompt position
        seq = ids
        for s in range(self.steps + 1):          # prefill tree + one tree per verify step
            seq = np.concatenate([seq, [[int(self.ea_layer.target_of[s % self.ea_layer.V])]]], axis=1)
            self.ea_layer.topK_genrate(hidden if s == 0 else np.zeros((1, 1, 1)),
                                       input_ids=seq, head=None, logits_processor=None)
        out = list(ids[0]) + [5] * 9 + [self.tok.eos_token_id]
        return [out], 10, self.steps - 1          # new_token=10 over `steps` verify passes


class TestRunnerIntegration(unittest.TestCase):
    def test_adaptive_records_delta_and_depth(self):
        from specdec.eagle_runner import EagleRunner, instrument_drafting
        layer = FakeDraftLayer(40, seed=11)
        layer.total_tokens = 19
        model = _FakeEaModelWithLayer(layer, steps=4)
        drafter = enable_adaptive(model, AdaptiveConfig(20, 0.03, 4), NumpyOps())
        instrument_drafting(model, static_depth=None)
        r = EagleRunner(model, sync=lambda: None, to_device=lambda ids: np.asarray([ids]),
                        seed_fn=lambda i: None)
        rec = r.generate([1, 2, 3], "spec")
        self.assertEqual(rec["tree_calls"], 5)
        self.assertEqual(rec["verify_steps"], 4)
        self.assertGreaterEqual(rec["draft_forwards"], 5)
        self.assertAlmostEqual(rec["delta"], rec["draft_forwards"] / 4)
        self.assertEqual(rec["mean_tree_nodes"], 20)         # every generated tree has N nodes
        self.assertGreaterEqual(rec["mean_tree_depth"], 1)
        # counters reset between generations
        rec2 = r.generate([1, 2, 3], "spec")
        self.assertEqual(rec2["tree_calls"], 5)

    def test_static_counts_one_plus_depth(self):
        from specdec.eagle_runner import EagleRunner, instrument_drafting
        layer = FakeDraftLayer(40, seed=12)

        def static_topk(hidden_states, input_ids, head, logits_processor):
            return (np.zeros((1, 8), dtype=np.int64), None, None, np.array([0, 1, 1, 2, 2, 3, 3, 3]))
        layer.topK_genrate = static_topk
        model = _FakeEaModelWithLayer(layer, steps=3)
        instrument_drafting(model, static_depth=5)
        r = EagleRunner(model, sync=lambda: None, to_device=lambda ids: np.asarray([ids]),
                        seed_fn=lambda i: None)
        rec = r.generate([1, 2], "spec")
        self.assertEqual(rec["draft_forwards"], 4 * 6)
        self.assertEqual(rec["mean_tree_depth"], 3)
        self.assertEqual(rec["mean_tree_nodes"], 8)

    def test_summarize_normalizes_legacy_adaptive_metadata(self):
        from scripts.summarize import describe

        meta = {
            "path": "eagle_repo",
            "mode": "spec",
            "use_eagle3": True,
            "tree": None,
            "previous_config": {"name": "old_n20_mu0.03_k4", "budget": 20, "mu": 0.03, "root_k": 4},
        }
        name, mode, config = describe(meta)

        self.assertEqual(name, "eagle3_adaptive_n20_mu0.03_k4")
        self.assertEqual(mode, "adaptive")
        self.assertEqual(config, {"num_draft_tokens": 20, "mu": 0.03, "root_k": 4})

    def test_summarize_names_adaptive_runs(self):
        import csv
        import subprocess
        import tempfile
        from specdec.eagle_runner import EagleRunner, instrument_drafting

        with tempfile.TemporaryDirectory() as d:
            qs = [{"bench": "gsm8k", "question_id": i, "turns": ["q"]} for i in range(5)]
            layer = FakeDraftLayer(40, seed=13)
            layer.total_tokens = 19
            model = _FakeEaModelWithLayer(layer, steps=3)
            cfg = AdaptiveConfig(20, 0.03, 4)
            enable_adaptive(model, cfg, NumpyOps())
            instrument_drafting(model, None)
            r = EagleRunner(model, sync=lambda: None, to_device=lambda ids: np.asarray([ids]),
                            seed_fn=lambda i: None)
            meta = {"path": "eagle_repo", "bench": "gsm8k", "mode": "spec", "batch_size": 1,
                    "temperature": 0.0, "use_eagle3": True, "tree": None, "adaptive": cfg.to_dict()}
            r.run(qs, "spec", os.path.join(d, "t.jsonl"), meta=meta, warmup=0)
            out = os.path.join(d, "s.csv")
            p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "summarize.py"), d,
                                "--out", out, "--n-boot", "100"], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            row = list(csv.DictReader(open(out)))[0]
            self.assertEqual(row["config"], "eagle3_adaptive_n20_mu0.03_k4")
            self.assertEqual(row["mode"], "adaptive")
            self.assertEqual(row["budget"], "20")
            self.assertTrue(float(row["delta"]) > 0 and float(row["mean_tree_depth"]) >= 1)


if __name__ == "__main__":
    unittest.main()
