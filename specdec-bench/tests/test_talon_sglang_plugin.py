"""Tests for TALON SGLang plugin.

Tests the plugin registration, patching mechanism, and marker file handling.
Run with: python -m unittest tests.test_talon_sglang_plugin -v
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specdec.talon_sglang import TalonSGLangConfig
from specdec.sglang_talon_patch import (
    TalonPatchState,
    _write_talon_marker,
    check_worker,
)


class TestPluginRegistration(unittest.TestCase):
    """Test that the plugin is properly registered."""

    def test_entry_point_registered(self):
        """Verify the plugin entry point is registered in pyproject.toml.

        Note: This test requires `pip install -e .` to register the plugin.
        It will skip if the package is not installed.
        """
        try:
            from importlib.metadata import entry_points
            eps = list(entry_points(group="sglang.srt.plugins"))
            names = [ep.name for ep in eps]
            values = [ep.value for ep in eps]

            if not names:
                self.skipTest(
                    "No plugins registered. Run `pip install -e .` to register the TALON plugin."
                )

            self.assertIn("talon", names, "TALON plugin not registered in entry points")
            self.assertIn("specdec.sglang_talon_patch:plugin", values)
        except ImportError:
            # Python < 3.10
            from importlib.metadata import entry_points as ep_func
            eps = ep_func()
            if hasattr(eps, "get"):
                sglang_eps = eps.get("sglang.srt.plugins", [])
            else:
                sglang_eps = [e for e in eps if e.group == "sglang.srt.plugins"]
            names = [ep.name for ep in sglang_eps]
            if not names:
                self.skipTest(
                    "No plugins registered. Run `pip install -e .` to register the TALON plugin."
                )
            self.assertIn("talon", names, "TALON plugin not registered")


class TestMarkerFile(unittest.TestCase):
    """Test marker file writing and verification."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = TalonSGLangConfig(budget=60, mu=0.03, root_k=10, num_steps=9)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_talon_marker(self):
        """Test marker file is written correctly."""
        _write_talon_marker(self.tmpdir, self.config)

        marker_path = os.path.join(self.tmpdir, "server.log.talon.jsonl")
        self.assertTrue(os.path.exists(marker_path))

        with open(marker_path) as f:
            line = f.readline()
            marker = json.loads(line)

        self.assertTrue(marker["patched"])
        self.assertEqual(marker["budget"], 60)
        self.assertAlmostEqual(marker["mu"], 0.03)
        self.assertEqual(marker["root_k"], 10)
        self.assertEqual(marker["num_steps"], 9)
        self.assertIn("timestamp", marker)

    def test_marker_appends(self):
        """Test multiple marker writes append to the file."""
        _write_talon_marker(self.tmpdir, self.config)
        _write_talon_marker(self.tmpdir, self.config)

        marker_path = os.path.join(self.tmpdir, "server.log.talon.jsonl")
        with open(marker_path) as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 2)


class TestCheckWorker(unittest.TestCase):
    """Test the check_worker function used by the launcher."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_check_worker_no_marker(self):
        """Test check_worker returns False when marker is missing."""
        class MockSpec:
            talon_config = TalonSGLangConfig(budget=60, mu=0.03, root_k=10)

        result = check_worker(MockSpec(), self.tmpdir)
        self.assertFalse(result)

    def test_check_worker_valid_marker(self):
        """Test check_worker returns True for valid marker."""
        config = TalonSGLangConfig(budget=60, mu=0.03, root_k=10, num_steps=9)
        _write_talon_marker(self.tmpdir, config)

        class MockSpec:
            talon_config = config

        result = check_worker(MockSpec(), self.tmpdir)
        self.assertTrue(result)

    def test_check_worker_patched_false(self):
        """Test check_worker returns False when patched=false."""
        marker_path = os.path.join(self.tmpdir, "server.log.talon.jsonl")
        with open(marker_path, "w") as f:
            f.write(json.dumps({"patched": False, "budget": 60}) + "\n")

        class MockSpec:
            talon_config = TalonSGLangConfig(budget=60, mu=0.03, root_k=10)

        result = check_worker(MockSpec(), self.tmpdir)
        self.assertFalse(result)

    def test_check_worker_config_mismatch(self):
        """Test check_worker returns False when config doesn't match."""
        config = TalonSGLangConfig(budget=60, mu=0.03, root_k=10, num_steps=9)
        _write_talon_marker(self.tmpdir, config)

        class MockSpec:
            # Different budget
            talon_config = TalonSGLangConfig(budget=32, mu=0.03, root_k=10)

        result = check_worker(MockSpec(), self.tmpdir)
        self.assertFalse(result)


class TestPatchState(unittest.TestCase):
    """Test TalonPatchState tracking."""

    def test_initial_state(self):
        """Test initial state is not patched."""
        state = TalonPatchState()
        self.assertFalse(state.patched)
        self.assertIsNone(state.original_select)
        self.assertEqual(state.layer_counter, 0)
        self.assertEqual(state.tree_counter, 0)

    def test_reset_tree(self):
        """Test reset_tree increments tree counter and resets layer."""
        state = TalonPatchState()
        state.layer_counter = 5
        state.reset_tree()
        self.assertEqual(state.layer_counter, 0)
        self.assertEqual(state.tree_counter, 1)


class TestSGLangSourcePins(unittest.TestCase):
    """Test that SGLang source code matches expected patterns.

    These tests verify that the SGLang functions we patch still exist
    and have the expected signatures. If these fail after an SGLang
    upgrade, the plugin may need updates.
    """

    @unittest.skipUnless(
        os.environ.get("SGLANG_REPO") or os.path.exists("third_party/sglang"),
        "SGLang not available"
    )
    def test_eagle_utils_exists(self):
        """Test eagle_utils module exists with expected function."""
        try:
            from sglang.srt.speculative import eagle_utils
            self.assertTrue(hasattr(eagle_utils, "select_top_k_tokens"),
                           "select_top_k_tokens not found in eagle_utils")
        except ImportError:
            self.skipTest("sglang not installed")

    @unittest.skipUnless(
        os.environ.get("SGLANG_REPO") or os.path.exists("third_party/sglang"),
        "SGLang not available"
    )
    def test_select_top_k_tokens_signature(self):
        """Test select_top_k_tokens has expected parameters."""
        try:
            from sglang.srt.speculative import eagle_utils
            import inspect
            sig = inspect.signature(eagle_utils.select_top_k_tokens)
            params = list(sig.parameters.keys())
            # Must have these parameters for our patch to work
            self.assertIn("logprobs", params[:3],
                         f"Expected 'logprobs' in first 3 params, got {params}")
        except ImportError:
            self.skipTest("sglang not installed")


if __name__ == "__main__":
    unittest.main()
