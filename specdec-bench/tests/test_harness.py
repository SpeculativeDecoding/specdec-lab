"""Unit and integration tests that run without a GPU or torch.

  python -m unittest discover -s tests -v
Source-pin tests run when EAGLE_REPO and SGLANG_REPO point at checkouts
(defaults: ~/EAGLE and ~/sglang) and are skipped otherwise.
"""
import ast
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from fakes import FakeEaModel, FakeTokenizer  # noqa: E402
from specdec import data, metrics  # noqa: E402
from specdec.eagle_runner import EagleRunner, read_jsonl  # noqa: E402
from specdec.sglang_client import run_load, server_step_times  # noqa: E402
from specdec.sglang_launcher import ServerSpec  # noqa: E402
from specdec.tree_config import PLAN_GRID, SGLANG_AUTO_LLAMA, TreeConfig, plan_grid  # noqa: E402

EAGLE_REPO = os.environ.get("EAGLE_REPO", os.path.expanduser("~/EAGLE"))
SGLANG_REPO = os.environ.get("SGLANG_REPO", os.path.expanduser("~/sglang"))
DATA_DIR = os.path.join(EAGLE_REPO, "eagle", "data")
HAVE_EAGLE = os.path.isdir(os.path.join(EAGLE_REPO, "eagle", "model"))
HAVE_SGLANG = os.path.isdir(os.path.join(SGLANG_REPO, "python", "sglang", "srt"))


# ----------------------------------------------------------------------------- tree configs
class TestTreeConfig(unittest.TestCase):
    def test_plan_grid_valid_and_pools(self):
        pools = {b: [t.pool for t in PLAN_GRID[b]] for b in PLAN_GRID}
        self.assertEqual(pools[16], [15, 26, 39, 36, 72])
        self.assertEqual(pools[32], [34, 48, 68, 114, 136])
        self.assertEqual(pools[60], [62, 84, 116, 186, 510, 810])
        self.assertEqual(len(plan_grid()), 16)
        for t in plan_grid():
            self.assertEqual(t.problems(), [], t.name)

    def test_invalid_pool(self):
        t = TreeConfig(8, 2, 32)            # pool 30 < 31 needed
        self.assertEqual(t.pool, 30)
        self.assertTrue(any("pool" in p for p in t.problems()))
        with self.assertRaises(ValueError):
            t.validate()

    def test_topk1_rule(self):
        self.assertEqual(TreeConfig(3, 1, 4).problems(), [])
        self.assertTrue(TreeConfig(3, 1, 8).problems())

    def test_mapping_to_eagle_repo(self):
        t = TreeConfig(6, 10, 60)
        # EAGLE eval-script defaults: --total-token 60 --depth 5 --top-k 10
        self.assertEqual(t.eagle_kwargs(), {"total_token": 60, "depth": 5, "top_k": 10})
        self.assertEqual(t.sglang_args(), ["--speculative-num-steps", "6",
                                           "--speculative-eagle-topk", "10",
                                           "--speculative-num-draft-tokens", "60"])
        self.assertEqual(SGLANG_AUTO_LLAMA.problems(), [])
        # adaptive tree's EAGLE-3 baseline (K=10, D=8, N=60) == EAGLE repo depth 8 == steps 9
        base = [t for t in PLAN_GRID[60] if t.label == "adaptive_eagle3_baseline"][0]
        self.assertEqual(base.eagle_kwargs(), {"total_token": 60, "depth": 8, "top_k": 10})


# ----------------------------------------------------------------------------- data
class TestData(unittest.TestCase):
    def test_adaptive_suite_entry_points(self):
        with open(os.path.join(ROOT, "scripts", "run_adaptive_suite.sh"), encoding="utf-8") as f:
            full_suite = f.read()
        with open(os.path.join(ROOT, "scripts", "run_adaptive_three_bench_suite.sh"), encoding="utf-8") as f:
            subset_suite = f.read()
        with open(os.path.join(ROOT, "chtc", "job_adaptive.sh"), encoding="utf-8") as f:
            chtc_job = f.read()

        self.assertIn("BENCHES=${BENCHES:-all}", full_suite)
        self.assertIn("export BENCHES=humaneval,alpaca,mt_bench", subset_suite)
        # the CHTC job runs the full suite on the benchmarks it is given (default: the three)
        self.assertIn('BENCHES=${1:-humaneval,alpaca,mt_bench}', chtc_job)
        self.assertIn('BENCHES="$BENCHES" bash scripts/run_adaptive_suite.sh', chtc_job)

    def test_parse_benches(self):
        self.assertEqual(data.parse_benches("all"), list(data.BENCHES))
        self.assertEqual(data.parse_benches("humaneval, alpaca,mt_bench"),
                         ["humaneval", "alpaca", "mt_bench"])
        for invalid in ("", "humaneval,unknown", "alpaca,alpaca"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                data.parse_benches(invalid)

    def test_encode_and_truncate(self):
        tok = FakeTokenizer()
        msgs = data.initial_messages() + [{"role": "user", "content": "hi"}]
        ids = data.encode_messages(tok, msgs)
        self.assertTrue(len(ids) > 10)
        self.assertEqual(data.stop_token_ids(tok), [128009])
        self.assertEqual(data.truncate_at_stop([1, 2, 128009, 3], [128009]), [1, 2])
        self.assertEqual(data.truncate_at_stop([1, 2], [128009]), [1, 2])

    def test_split_deterministic(self):
        qs = [{"bench": "x", "question_id": i, "turns": ["a"]} for i in range(1000)]
        c1, t1 = data.split(qs)
        c2, t2 = data.split(list(reversed(qs)))
        self.assertEqual({q["question_id"] for q in c1}, {q["question_id"] for q in c2})
        self.assertEqual(len(c1) + len(t1), 1000)
        self.assertTrue(250 < len(c1) < 350, len(c1))

    @unittest.skipUnless(os.path.isdir(DATA_DIR), "EAGLE data not present")
    def test_eagle_question_files(self):
        counts = {b: len(data.load_questions(DATA_DIR, b)) for b in data.BENCHES}
        self.assertEqual(counts, {"mt_bench": 80, "humaneval": 80, "gsm8k": 80,
                                  "alpaca": 80, "sum": 80, "qa": 80})
        mt = data.load_questions(DATA_DIR, "mt_bench")
        self.assertTrue(all(len(q["turns"]) == 2 for q in mt))


# ----------------------------------------------------------------------------- metrics
class TestMetrics(unittest.TestCase):
    def test_tau_and_throughput(self):
        rows = [{"new_tokens": 10, "verify_steps": 4, "output_tokens": 9, "wall_time": 1.0},
                {"new_tokens": 6, "verify_steps": 4, "output_tokens": 6, "wall_time": 2.0}]
        self.assertAlmostEqual(metrics.pooled_tau(rows), 2.0)
        self.assertAlmostEqual(metrics.throughput(rows), 5.0)
        self.assertAlmostEqual(metrics.geometric_mean([1, 4]), 2.0)

    def test_bootstrap(self):
        sd = [{"qid_turn": i, "output_tokens": 100, "wall_time": 1.0 + 0.1 * (i % 3)} for i in range(50)]
        ar = [{"qid_turn": i, "output_tokens": 100, "wall_time": 3.0} for i in range(50)]
        ci = metrics.paired_bootstrap_speedup(sd, ar, n_boot=2000)
        self.assertTrue(ci["lo"] <= ci["speedup"] <= ci["hi"])
        self.assertAlmostEqual(ci["speedup"], 3.0 / (sum(1.0 + 0.1 * (i % 3) for i in range(50)) / 50), 6)

    def test_full_occupancy_window(self):
        # 2 slots: [0,4] & [0,2] then [2,5]; occupancy 2 from t=0 to t=4
        spans = [{"start": 0, "end": 4, "output_tokens": 40},
                 {"start": 0, "end": 2, "output_tokens": 20},
                 {"start": 2, "end": 5, "output_tokens": 30}]
        self.assertEqual(metrics.full_occupancy_window(spans, 2), (0, 4))
        tp = metrics.overlap_throughput(spans, 0, 4)
        self.assertAlmostEqual(tp, (40 + 20 + 30 * 2 / 3) / 4)
        self.assertIsNone(metrics.full_occupancy_window(spans, 3))


# ----------------------------------------------------------------------------- EAGLE runner
class TestEagleRunner(unittest.TestCase):
    def make(self, **kw):
        tok = FakeTokenizer()
        model = FakeEaModel(tok, **kw)
        r = EagleRunner(model, max_new_tokens=128, max_length=4096,
                        sync=lambda: None, to_device=lambda ids: [ids], seed_fn=lambda i: None)
        return r, model

    def test_generate_spec(self):
        r, model = self.make(n_tokens=20, accept_per_step=3, extra_after_stop=2)
        out = r.generate([1, 2, 3], "spec")
        self.assertEqual(out["output_tokens"], 20)          # truncated before EOS
        self.assertTrue(out["hit_stop"])
        self.assertEqual(out["new_tokens"], 23)
        self.assertEqual(out["verify_steps"], 6)              # ceil(23/4)
        self.assertAlmostEqual(out["tau"], 23 / 6)
        kw = model.calls[-1]
        self.assertEqual(kw["max_new_tokens"], 128)
        self.assertEqual(kw["max_length"], 4096)
        self.assertTrue(kw["is_llama3"])

    def test_multiturn_and_file(self):
        r, _ = self.make()
        q = {"bench": "mt_bench", "question_id": 81, "turns": ["one", "two"]}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "o.jsonl")
            rows = r.run([q], "ar", path, meta={"x": 1}, warmup=1)
            self.assertEqual([x["turn"] for x in rows], [0, 1])
            self.assertGreater(rows[1]["prompt_tokens"], rows[0]["prompt_tokens"])  # answer fed back
            self.assertAlmostEqual(rows[0]["tau"], 1.0)
            meta, back = read_jsonl(path)
            self.assertEqual(meta, {"x": 1})
            self.assertEqual(len(back), 2)

    def test_interrupted_run_leaves_no_result(self):
        # A run killed mid-way (Colab disconnect) must not leave a file that
        # --skip-existing would treat as finished.
        r, _ = self.make()
        qs = [{"bench": "mt_bench", "question_id": i, "turns": ["hi"]} for i in range(3)]
        real, calls = r.run_question, []

        def flaky(q, mode, seed=0):
            calls.append(q["question_id"])
            if len(calls) == 2:
                raise KeyboardInterrupt
            return real(q, mode, seed)

        r.run_question = flaky
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "o.jsonl")
            with self.assertRaises(KeyboardInterrupt):
                r.run(qs, "ar", path, warmup=0)
            self.assertFalse(os.path.exists(path))
            self.assertTrue(os.path.exists(path + ".partial"))
            r.run_question = real
            r.run(qs, "ar", path, warmup=0)                  # re-run completes and renames
            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".partial"))
            self.assertEqual(len(read_jsonl(path)[1]), 3)

    def _qs(self, n, turns=1):
        return [{"bench": "mt_bench", "question_id": i, "turns": [f"t{j}" for j in range(turns)]}
                for i in range(n)]

    def _interrupt_after(self, r, n_ok, mid_turn=False):
        """Make run_question fail on question n_ok (optionally after writing nothing)."""
        real, seen = r.run_question, []

        def flaky(q, mode, seed=0):
            seen.append(q["question_id"])
            if q["question_id"] == n_ok and not getattr(r, "_warm", False):
                raise KeyboardInterrupt
            return real(q, mode, seed)
        r.run_question = flaky
        return real, seen

    @staticmethod
    def _strip(rows):
        return [{k: v for k, v in x.items() if k != "wall_time"} for x in rows]

    def test_resume_runs_only_missing_questions(self):
        r, _ = self.make()
        qs = self._qs(5, turns=2)
        meta = {"tree": "s9_k10_b60", "gpu": "A100"}
        with tempfile.TemporaryDirectory() as d:
            clean = os.path.join(d, "clean.jsonl")
            r.run(qs, "spec", clean, meta=meta, warmup=0)
            path = os.path.join(d, "o.jsonl")
            real, _ = self._interrupt_after(r, 3)
            with self.assertRaises(KeyboardInterrupt):
                r.run(qs, "spec", path, meta=meta, warmup=0)
            self.assertFalse(os.path.exists(path))
            # simulate a crash in the middle of writing a line
            with open(path + ".partial", "a") as f:
                f.write('{"bench": "mt_bench", "question_id": 3, "tu')
            r.run_question = real
            calls = []
            r.run_question = lambda q, m, seed=0: (calls.append(q["question_id"]), real(q, m, seed))[1]
            rows = r.run(qs, "spec", path, meta=meta, warmup=1)
            self.assertEqual(calls, [3, 3, 4])                  # 1 warm-up + the 2 missing questions
            self.assertFalse(os.path.exists(path + ".partial"))
            m1, a = read_jsonl(clean)
            m2, b = read_jsonl(path)
            self.assertEqual(m1, m2)
            self.assertEqual(self._strip(a), self._strip(b))     # same rows, same order
            self.assertEqual(self._strip(rows), self._strip(b))

    def test_resume_drops_incomplete_multiturn_question(self):
        r, _ = self.make()
        qs = self._qs(3, turns=2)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "o.jsonl")
            r.run(qs[:1], "ar", path, meta={"a": 1}, warmup=0)
            os.replace(path, path + ".partial")
            _, rows = read_jsonl(path + ".partial")
            with open(path + ".partial", "a") as f:             # question 1: only turn 0 written
                f.write(json.dumps(dict(rows[0], question_id=1)) + "\n")
            calls = []
            real = r.run_question
            r.run_question = lambda q, m, seed=0: (calls.append(q["question_id"]), real(q, m, seed))[1]
            r.run(qs, "ar", path, meta={"a": 1}, warmup=0)
            self.assertEqual(calls, [1, 2])
            _, back = read_jsonl(path)
            self.assertEqual([(x["question_id"], x["turn"]) for x in back],
                             [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])

    def test_resume_ignores_checkpoint_from_other_settings(self):
        r, _ = self.make()
        qs = self._qs(3)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "o.jsonl")
            r.run(qs, "ar", path, meta={"gpu": "L4"}, warmup=0)
            os.replace(path, path + ".partial")
            calls = []
            real = r.run_question
            r.run_question = lambda q, m, seed=0: (calls.append(q["question_id"]), real(q, m, seed))[1]
            r.run(qs, "ar", path, meta={"gpu": "A100"}, warmup=0)
            self.assertEqual(calls, [0, 1, 2])
            meta, back = read_jsonl(path)
            self.assertEqual(meta, {"gpu": "A100"})
            self.assertEqual(len(back), 3)
            # and resume=False always starts over
            os.replace(path, path + ".partial")
            calls.clear()
            r.run(qs, "ar", path, meta={"gpu": "A100"}, warmup=0, resume=False)
            self.assertEqual(calls, [0, 1, 2])

    def test_skip_existing_flag(self):
        src = open(os.path.join(ROOT, "scripts", "run_eagle.py")).read()
        self.assertIn('"--skip-existing"', src)
        self.assertRegex(src, r"if args\.skip_existing and os\.path\.exists\(out\):\s+.*\n\s+continue")
        self.assertIn("--skip-existing", open(os.path.join(ROOT, "scripts", "run_adaptive_suite.sh")).read())

    def test_length_risk_flag(self):
        r, _ = self.make()
        self.assertTrue(r.generate(list(range(4000)), "spec")["length_risk"])
        self.assertFalse(r.generate(list(range(100)), "spec")["length_risk"])


# ----------------------------------------------------------------------------- SGLang client (mock server)
class _MockState:
    lock = threading.Lock()
    active = 0
    max_active = 0
    requests = 0


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/server_info":
            body = {"internal_states": [{"step_time_dict": {"2": [0.010, 0.012, 0.011, 0.020]}}]}
        else:
            body = {}
        self._send(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/flush_cache":
            return self._send({})
        assert "input_ids" in payload and "sampling_params" in payload
        with _MockState.lock:
            _MockState.active += 1
            _MockState.requests += 1
            _MockState.max_active = max(_MockState.max_active, _MockState.active)
        time.sleep(0.05)
        with _MockState.lock:
            _MockState.active -= 1
        self._send({"text": " answer ", "meta_info": {
            "completion_tokens": 40, "prompt_tokens": len(payload["input_ids"]),
            "finish_reason": {"type": "stop"}, "spec_verify_ct": 10, "spec_accept_length": 4.0}})

    def _send(self, body):
        b = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class TestSGLangClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.url = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def test_concurrency_and_summary(self):
        _MockState.max_active = 0
        _MockState.requests = 0
        qs = [{"bench": "gsm8k", "question_id": i, "turns": ["q"]} for i in range(12)]
        qs.append({"bench": "mt_bench", "question_id": 99, "turns": ["a", "b"]})
        res = run_load(self.url, FakeTokenizer(), qs, concurrency=2, max_new_tokens=64)
        s = res["summary"]
        self.assertEqual(s["n_errors"], 0)
        self.assertEqual(s["n_turns"], 14)
        self.assertEqual(_MockState.requests, 14)
        self.assertLessEqual(_MockState.max_active, 2)
        self.assertEqual(_MockState.max_active, 2)
        self.assertAlmostEqual(s["tau"], 4.0)
        self.assertIsNotNone(s["throughput_full_occupancy"])
        mt = [r for r in res["rows"] if r["bench"] == "mt_bench"]
        self.assertGreater(mt[1]["prompt_tokens"], mt[0]["prompt_tokens"])

    def test_step_times(self):
        st = server_step_times(self.url, 2)
        self.assertEqual(st["n"], 4)
        self.assertAlmostEqual(st["median"], 0.0115)
        self.assertIsNone(server_step_times(self.url, 8))


# ----------------------------------------------------------------------------- launcher + sweep
class TestLauncher(unittest.TestCase):
    def test_commands(self):
        spec = ServerSpec(mode="spec", tree=TreeConfig(6, 10, 60))
        cmd = spec.command("python")
        self.assertIn("--cuda-graph-max-bs-decode", cmd)
        self.assertIn("--disable-radix-cache", cmd)
        self.assertEqual(cmd[cmd.index("--speculative-algorithm") + 1], "EAGLE3")
        ar = ServerSpec(mode="ar").command("python")
        self.assertNotIn("--speculative-algorithm", ar)
        auto = ServerSpec(mode="auto").command("python")
        self.assertNotIn("--speculative-num-steps", auto)
        ad = ServerSpec(mode="adaptive").command("python")
        self.assertIn("--speculative-adaptive", ad)
        with self.assertRaises(ValueError):
            ServerSpec(mode="adaptive", adaptive_start=TreeConfig(3, 4, 16)).command("python")

    def test_sweep_dry_run(self):
        for cfg, n in (("sweep_eagle3_llama31.yaml", 21), ("smoke.yaml", 2)):
            out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_sglang_sweep.py"),
                                  "--config", os.path.join(ROOT, "configs", cfg),
                                  "--data-dir", "/nonexistent", "--out-dir", "/tmp/x", "--dry-run"],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(len([l for l in out.stdout.splitlines() if " : " in l]), n, out.stdout)


# ----------------------------------------------------------------------------- end-to-end summarize
class TestSummarize(unittest.TestCase):
    def test_summarize_eagle_files(self):
        tok = FakeTokenizer()
        with tempfile.TemporaryDirectory() as d:
            qs = [{"bench": "gsm8k", "question_id": i, "turns": ["q"]} for i in range(8)]
            for mode, k in (("spec", 3), ("ar", 0)):
                r = EagleRunner(FakeEaModel(tok, accept_per_step=k), sync=lambda: None,
                                to_device=lambda ids: [ids], seed_fn=lambda i: None)
                meta = {"path": "eagle_repo", "bench": "gsm8k", "mode": mode, "batch_size": 1,
                        "temperature": 0.0, "use_eagle3": True,
                        "tree": TreeConfig(6, 10, 60).to_dict() if mode == "spec" else None}
                r.run(qs, mode, os.path.join(d, f"gsm8k__{mode}.jsonl"), meta=meta, warmup=0)
            out = os.path.join(d, "s.csv")
            p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "summarize.py"), d,
                                "--out", out, "--n-boot", "200"], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            rows = list(csv.DictReader(open(out)))
            spec = [x for x in rows if x["mode"] == "spec"][0]
            self.assertEqual(spec["config"], "eagle3_s6_k10_b60")
            self.assertTrue(spec["speedup"] and spec["speedup_ci_lo"])
            lossless = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check_lossless.py"),
                                       os.path.join(d, "gsm8k__spec.jsonl"), os.path.join(d, "gsm8k__ar.jsonl")],
                                      capture_output=True, text=True)
            self.assertIn("8 identical", lossless.stdout)


# ----------------------------------------------------------------------------- source pins
@unittest.skipUnless(HAVE_EAGLE, "EAGLE checkout not found (set EAGLE_REPO)")
class TestPinsEagle(unittest.TestCase):
    def _func(self, path, name):
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        self.fail(f"{name} not found in {path}")

    def test_signatures(self):
        p = os.path.join(EAGLE_REPO, "eagle", "model", "ea_model.py")
        fp = self._func(p, "from_pretrained")
        names = [a.arg for a in fp.args.args]
        for n in ("use_eagle3", "base_model_path", "ea_model_path", "total_token", "depth", "top_k"):
            self.assertIn(n, names)
        self.assertIsNotNone(fp.args.kwarg)          # **kwargs forwarded to HF from_pretrained
        for fn in ("eagenerate", "naivegenerate"):
            args = [a.arg for a in self._func(p, fn).args.args]
            for n in ("input_ids", "temperature", "max_new_tokens", "max_length", "log", "is_llama3"):
                self.assertIn(n, args, f"{fn}.{n}")
        src = open(p).read()
        self.assertIn("return input_ids, new_token, idx", src)
        self.assertIn("initialize_past_key_values(self.base_model,max_length=max_length)", src)
        self.assertIn('if hasattr(self, "past_key_values")', src)   # cache allocated once

    def test_tree_pool_and_counting(self):
        for f in ("cnets.py", "cnets1.py"):   # EAGLE-3 and EAGLE-1 draft layers
            c = open(os.path.join(EAGLE_REPO, "eagle", "model", f)).read()
            self.assertIn("self.total_tokens = total_tokens - 1", c, f)
            self.assertIn("for i in range(depth):", c, f)
            self.assertIn("scores_list.append(cu_scores)", c, f)
            self.assertIn("top_scores = torch.topk(scores_list, total_tokens, dim=-1)", c, f)
        u = open(os.path.join(EAGLE_REPO, "eagle", "model", "utils.py")).read()
        self.assertIn("new_token += accept_length + 1", u)

    def test_system_prompt_verbatim(self):
        p = os.path.join(EAGLE_REPO, "eagle", "evaluation", "gen_ea_answer_llama3chat.py")
        found = set()
        for node in ast.walk(ast.parse(open(p).read())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("You are a helpful"):
                found.add(node.value)
        self.assertEqual(found, {data.SYSTEM_PROMPT})

    def test_transformers_import_needs_losskwargs(self):
        q = open(os.path.join(EAGLE_REPO, "eagle", "model", "modeling_qwen3_kv.py")).read()
        self.assertIn("LossKwargs", q)   # reason for pinning transformers==4.53.3
        self.assertIn("modeling_qwen3_kv", open(os.path.join(EAGLE_REPO, "eagle", "model", "ea_model.py")).read())


@unittest.skipUnless(HAVE_SGLANG, "SGLang checkout not found (set SGLANG_REPO)")
class TestPinsSGLang(unittest.TestCase):
    def test_cli_flags_exist(self):
        srt = os.path.join(SGLANG_REPO, "python", "sglang", "srt")
        fields = ""
        for root in (os.path.join(srt, "arg_groups", "fields"),):
            if os.path.isdir(root):
                for f in os.listdir(root):
                    fields += open(os.path.join(root, f)).read()
        sa = os.path.join(srt, "server_args.py")
        fields += open(sa).read()
        spec = ServerSpec(mode="spec", tree=TreeConfig(6, 10, 60), adaptive_config="x.json")
        flags = set(x for x in spec.command("python") if x.startswith("--"))
        flags |= {"--speculative-adaptive", "--speculative-adaptive-config"}
        for flag in sorted(flags):
            field = flag[2:].replace("-", "_")
            self.assertTrue(re.search(rf"\b{field}\s*:", fields),
                            f"{flag} is not a server arg in this SGLang checkout")

    def test_tree_semantics(self):
        srt = os.path.join(SGLANG_REPO, "python", "sglang", "srt")
        eu = open(os.path.join(srt, "speculative", "eagle_utils.py")).read()
        self.assertIn("top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)", eu)
        self.assertIn("n = 1 + (num_steps-1) * topk", eu)
        hook = open(os.path.join(srt, "arg_groups", "speculative_hook.py")).read()
        self.assertIn("speculative_num_draft_tokens=cfg.speculative_num_steps + 1", hook)
        self.assertRegex(hook, r'"LlamaForCausalLM"\]:\s*\n\s*return \(5, 4, 8\)')

    def test_meta_info_and_step_time(self):
        srt = os.path.join(SGLANG_REPO, "python", "sglang", "srt")
        tm = open(os.path.join(srt, "managers", "tokenizer_manager.py")).read()
        self.assertIn('meta_info["spec_verify_ct"]', tm)
        self.assertIn("recv_obj.completion_tokens[i] / recv_obj.spec_verify_ct[i]", tm)
        self.assertIn('"completion_tokens": recv_obj.completion_tokens[i]', tm)
        found = subprocess.run(["grep", "-rl", "step_time_dict", os.path.join(srt, "managers")],
                               capture_output=True, text=True).stdout
        self.assertTrue(found.strip())


class TestScriptsCompile(unittest.TestCase):
    def test_compile(self):
        import py_compile
        for d in ("specdec", "scripts", "tests"):
            for f in os.listdir(os.path.join(ROOT, d)):
                if f.endswith(".py"):
                    py_compile.compile(os.path.join(ROOT, d, f), doraise=True)


if __name__ == "__main__":
    unittest.main()
