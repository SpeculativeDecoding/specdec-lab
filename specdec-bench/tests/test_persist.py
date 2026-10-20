"""Round-trip tests for scripts/persist.py (model cache, code, environments)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "persist.py")
REPO = "meta-llama/Llama-3.1-8B-Instruct"
NAME = "models--meta-llama--Llama-3.1-8B-Instruct"


def run(args, hub, check=True):
    env = dict(os.environ, HF_HUB_CACHE=hub, HF_HUB_OFFLINE="1")
    p = subprocess.run([sys.executable, SCRIPT] + args, env=env, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise AssertionError(p.stdout + p.stderr)
    return p


def fake_hf_entry(hub):
    """Same layout as the real HF cache: blobs + snapshot files as relative symlinks."""
    base = os.path.join(hub, NAME)
    os.makedirs(os.path.join(base, "blobs"))
    os.makedirs(os.path.join(base, "snapshots", "abc123"))
    os.makedirs(os.path.join(base, "refs"))
    open(os.path.join(base, "refs", "main"), "w").write("abc123")
    for fname, blob, data in [("config.json", "b1", b'{"x": 1}'), ("model.safetensors", "b2", os.urandom(4096))]:
        open(os.path.join(base, "blobs", blob), "wb").write(data)
        os.symlink(os.path.join("..", "..", "blobs", blob), os.path.join(base, "snapshots", "abc123", fname))
    return base


class TestPersist(unittest.TestCase):
    def test_models_roundtrip_keeps_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            hub, store, hub2 = (os.path.join(d, x) for x in ("hub", "store", "hub2"))
            base = fake_hf_entry(hub)
            out = run(["save", "--store", store, "--what", "models", "--repos", REPO, "--no-download"], hub).stdout
            self.assertIn("saved", out)
            tar = os.path.join(store, "models", NAME + ".tar")
            self.assertTrue(os.path.exists(tar) and not os.path.exists(tar + ".partial"))
            self.assertEqual(json.load(open(tar + ".json"))["repo"], REPO)
            self.assertIn("already saved", run(["save", "--store", store, "--what", "models",
                                                "--repos", REPO, "--no-download"], hub).stdout)
            run(["restore", "--store", store, "--what", "models", "--repos", REPO], hub2)
            snap = os.path.join(hub2, NAME, "snapshots", "abc123", "model.safetensors")
            self.assertTrue(os.path.islink(snap))
            self.assertEqual(open(snap, "rb").read(),
                             open(os.path.join(base, "blobs", "b2"), "rb").read())
            self.assertIn("already in", run(["restore", "--store", store, "--what", "models",
                                             "--repos", REPO], hub2).stdout)

    def test_missing_model_fails_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            p = run(["save", "--store", os.path.join(d, "s"), "--what", "models", "--repos", REPO,
                     "--no-download"], os.path.join(d, "hub"), check=False)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("not in the HF cache", p.stdout + p.stderr)

    def test_code_roundtrip_excludes_heavy_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "specdec-bench")
            for rel, txt in [("specdec/a.py", "A"), ("README.md", "R"), (".venv-eagle/bin/x", "V"),
                             ("third_party/EAGLE/y", "E"), ("results/r.jsonl", "J"),
                             ("specdec/__pycache__/a.pyc", "C"), ("results.jsonl.partial", "P")]:
                os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                open(os.path.join(root, rel), "w").write(txt)
            store = os.path.join(d, "store")
            run(["save", "--store", store, "--what", "code", "--root", root], os.path.join(d, "hub"))
            import zipfile
            names = sorted(zipfile.ZipFile(os.path.join(store, "code", "specdec-bench.zip")).namelist())
            self.assertEqual(names, ["specdec-bench/README.md", "specdec-bench/specdec/a.py"])
            run(["save", "--store", store, "--what", "code", "--root", root], os.path.join(d, "hub"))
            self.assertTrue(os.path.exists(os.path.join(store, "code", "specdec-bench.prev.zip")))
            new_root = os.path.join(d, "fresh")
            run(["restore", "--store", store, "--what", "code", "--root", new_root], os.path.join(d, "hub"))
            self.assertEqual(open(os.path.join(new_root, "specdec", "a.py")).read(), "A")

    def test_envs_restore_only_into_same_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "specdec-bench")
            os.makedirs(os.path.join(root, ".venv-eagle", "bin"))
            os.symlink(sys.executable, os.path.join(root, ".venv-eagle", "bin", "python"))  # absolute link
            os.makedirs(os.path.join(root, "third_party", "EAGLE"))
            open(os.path.join(root, "third_party", "EAGLE", "setup.py"), "w").write("#")
            store, hub = os.path.join(d, "store"), os.path.join(d, "hub")
            run(["save", "--store", store, "--what", "envs", "--envs", "eagle", "--root", root], hub)
            other = os.path.join(d, "elsewhere")
            os.makedirs(other)
            out = run(["restore", "--store", store, "--what", "envs", "--envs", "eagle", "--root", other], hub).stdout
            self.assertIn("skipping", out)
            self.assertFalse(os.path.exists(os.path.join(other, ".venv-eagle")))
            import shutil
            shutil.rmtree(os.path.join(root, ".venv-eagle"))
            shutil.rmtree(os.path.join(root, "third_party"))
            out = run(["restore", "--store", store, "--what", "envs", "--envs", "eagle", "--root", root], hub).stdout
            self.assertIn("restored", out)
            self.assertEqual(os.readlink(os.path.join(root, ".venv-eagle", "bin", "python")), sys.executable)
            self.assertTrue(os.path.exists(os.path.join(root, "third_party", "EAGLE", "setup.py")))


if __name__ == "__main__":
    unittest.main()
