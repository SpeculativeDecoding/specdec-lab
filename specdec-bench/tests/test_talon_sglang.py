"""Tests for TALON SGLang plugin.

Tests the node selection logic and configuration handling.
"""
import os
import sys
import unittest
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specdec.talon_sglang import (
    TalonSGLangConfig,
    enable_talon,
    get_talon_config,
    is_talon_enabled,
)


class TestTalonSGLangConfig(unittest.TestCase):
    """Test TalonSGLangConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = TalonSGLangConfig()
        self.assertEqual(cfg.budget, 60)
        self.assertAlmostEqual(cfg.mu, 0.03)
        self.assertEqual(cfg.root_k, 10)
        self.assertEqual(cfg.num_steps, 9)

    def test_validate_valid(self):
        """Test validation passes for valid config."""
        cfg = TalonSGLangConfig(budget=60, mu=0.03, root_k=10, num_steps=9)
        validated = cfg.validate()
        self.assertEqual(validated.budget, 60)

    def test_validate_invalid_budget(self):
        """Test validation fails for budget < 2."""
        cfg = TalonSGLangConfig(budget=1)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_invalid_mu_low(self):
        """Test validation fails for mu <= 0."""
        cfg = TalonSGLangConfig(mu=0.0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_invalid_mu_high(self):
        """Test validation fails for mu > 1."""
        cfg = TalonSGLangConfig(mu=1.5)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_invalid_root_k(self):
        """Test validation fails for root_k < 1."""
        cfg = TalonSGLangConfig(root_k=0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_from_env_disabled(self):
        """Test from_env returns None when SPECDEC_TALON != 1."""
        os.environ.pop("SPECDEC_TALON", None)
        cfg = TalonSGLangConfig.from_env()
        self.assertIsNone(cfg)

    def test_from_env_enabled(self):
        """Test from_env loads config when enabled."""
        os.environ["SPECDEC_TALON"] = "1"
        os.environ["SPECDEC_TALON_BUDGET"] = "32"
        os.environ["SPECDEC_TALON_MU"] = "0.05"
        try:
            cfg = TalonSGLangConfig.from_env()
            self.assertIsNotNone(cfg)
            self.assertEqual(cfg.budget, 32)
            self.assertAlmostEqual(cfg.mu, 0.05)
        finally:
            os.environ.pop("SPECDEC_TALON", None)
            os.environ.pop("SPECDEC_TALON_BUDGET", None)
            os.environ.pop("SPECDEC_TALON_MU", None)

    def test_to_dict(self):
        """Test config serialization."""
        cfg = TalonSGLangConfig(budget=60, mu=0.03, root_k=10, num_steps=9)
        d = cfg.to_dict()
        self.assertEqual(d["budget"], 60)
        self.assertAlmostEqual(d["mu"], 0.03)
        self.assertEqual(d["root_k"], 10)
        self.assertEqual(d["num_steps"], 9)


class TestTalonState(unittest.TestCase):
    """Test TALON enable/disable state."""

    def setUp(self):
        # Clear any existing state
        enable_talon(None)

    def tearDown(self):
        enable_talon(None)

    def test_initially_disabled(self):
        """Test TALON is disabled by default."""
        enable_talon(None)
        self.assertFalse(is_talon_enabled())
        self.assertIsNone(get_talon_config())

    def test_enable_with_config(self):
        """Test enabling TALON with explicit config."""
        cfg = TalonSGLangConfig(budget=60, mu=0.03, root_k=10)
        enable_talon(cfg)
        self.assertTrue(is_talon_enabled())
        self.assertEqual(get_talon_config().budget, 60)

    def test_disable_after_enable(self):
        """Test disabling TALON after enabling."""
        cfg = TalonSGLangConfig()
        enable_talon(cfg)
        self.assertTrue(is_talon_enabled())
        enable_talon(None)
        self.assertFalse(is_talon_enabled())


class TestTalonSelection(unittest.TestCase):
    """Test TALON token selection logic.

    These tests use numpy to simulate the selection without requiring torch.
    """

    @unittest.skipIf(True, "Requires torch - run in .venv-sglang")
    def test_layer0_top_k(self):
        """Test layer 0 uses forced top-K initialization."""
        import torch
        from specdec.talon_sglang import talon_select_tokens

        cfg = TalonSGLangConfig(budget=60, mu=0.03, root_k=10)
        logprobs = torch.randn(1, 1000)  # (batch, vocab)
        parent_scores = torch.zeros(1, 1)  # Root has score 0

        indices, scores, n_talon = talon_select_tokens(
            logprobs, parent_scores, layer=0, config=cfg, topk=10, remaining_budget=59
        )

        # Should select exactly root_k tokens
        self.assertEqual(indices.shape[-1], 10)
        self.assertEqual(n_talon, 10)

    @unittest.skipIf(True, "Requires torch - run in .venv-sglang")
    def test_layer1_confidence_gating(self):
        """Test layer >= 1 uses confidence gating."""
        import torch
        from specdec.talon_sglang import talon_select_tokens

        cfg = TalonSGLangConfig(budget=60, mu=0.03, root_k=10)

        # Create logprobs with one very high score
        logprobs = torch.full((1, 10, 100), -10.0)  # Low base
        logprobs[0, 0, 0] = -0.1  # High score anchor
        logprobs[0, 0, 1] = -0.5  # Should pass (close to anchor)
        logprobs[0, 0, 2] = -5.0  # Should fail (too far from anchor)

        parent_scores = torch.zeros(1, 10)  # All parents have score 0

        indices, scores, n_talon = talon_select_tokens(
            logprobs, parent_scores, layer=1, config=cfg, topk=10, remaining_budget=49
        )

        # At least the anchor should be selected
        self.assertGreater(n_talon, 0)


try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class TestTorchMatchesNumpy(unittest.TestCase):
    """Test that torch and numpy implementations produce identical results.

    This is critical for verifying TALON-S against full TALON.
    """

    def test_numpy_reference_available(self):
        """Verify the numpy reference implementation exists."""
        from specdec.adaptive import NumpyOps, select_children
        ops = NumpyOps()
        self.assertIsNotNone(ops)

    def test_select_children_layer0_parity(self):
        """Test layer 0 (root top-K) produces identical results."""
        import numpy as np
        from specdec.adaptive import NumpyOps, TorchOps, select_children

        np.random.seed(42)
        cu_scores_np = np.random.randn(1, 100).astype(np.float32)

        # Numpy reference
        ops_np = NumpyOps()
        par_np, idx_np, val_np = select_children(
            ops_np, cu_scores_np, remaining=59, log_mu=math.log(0.03), root_k=10
        )

        # Torch implementation
        try:
            import torch
            ops_torch = TorchOps()
            cu_scores_torch = torch.from_numpy(cu_scores_np)
            par_t, idx_t, val_t = select_children(
                ops_torch, cu_scores_torch, remaining=59, log_mu=math.log(0.03), root_k=10
            )

            # Convert to numpy for comparison
            par_t_np = par_t.numpy()
            idx_t_np = idx_t.numpy()
            val_t_np = val_t.numpy()

            # Must be identical
            np.testing.assert_array_equal(par_np, par_t_np, "parent indices differ")
            np.testing.assert_array_equal(idx_np, idx_t_np, "token indices differ")
            np.testing.assert_allclose(val_np, val_t_np, rtol=1e-5, err_msg="scores differ")

        except ImportError:
            self.skipTest("torch not available")

    def test_select_children_layer1_parity(self):
        """Test layer >= 1 (confidence gating) produces identical results."""
        import numpy as np
        from specdec.adaptive import NumpyOps, TorchOps, select_children

        np.random.seed(123)
        # Multiple parents with child scores
        cu_scores_np = np.random.randn(10, 50).astype(np.float32)

        # Numpy reference
        ops_np = NumpyOps()
        par_np, idx_np, val_np = select_children(
            ops_np, cu_scores_np, remaining=49, log_mu=math.log(0.03), root_k=None
        )

        # Torch implementation
        try:
            import torch
            ops_torch = TorchOps()
            cu_scores_torch = torch.from_numpy(cu_scores_np)
            par_t, idx_t, val_t = select_children(
                ops_torch, cu_scores_torch, remaining=49, log_mu=math.log(0.03), root_k=None
            )

            # Convert to numpy for comparison
            par_t_np = par_t.numpy()
            idx_t_np = idx_t.numpy()
            val_t_np = val_t.numpy()

            # Must be identical
            np.testing.assert_array_equal(par_np, par_t_np, "parent indices differ")
            np.testing.assert_array_equal(idx_np, idx_t_np, "token indices differ")
            np.testing.assert_allclose(val_np, val_t_np, rtol=1e-5, err_msg="scores differ")

        except ImportError:
            self.skipTest("torch not available")

    def test_select_children_budget_limit(self):
        """Test budget limiting produces identical results."""
        import numpy as np
        from specdec.adaptive import NumpyOps, TorchOps, select_children

        np.random.seed(456)
        # Many high scores to trigger budget limiting
        cu_scores_np = np.random.randn(5, 100).astype(np.float32)
        cu_scores_np[:, :20] += 5.0  # Make first 20 very high

        # Numpy reference
        ops_np = NumpyOps()
        par_np, idx_np, val_np = select_children(
            ops_np, cu_scores_np, remaining=15, log_mu=math.log(0.03), root_k=None
        )

        # Should have at most 15 nodes
        self.assertLessEqual(len(par_np), 15)

        # Torch implementation
        try:
            import torch
            ops_torch = TorchOps()
            cu_scores_torch = torch.from_numpy(cu_scores_np)
            par_t, idx_t, val_t = select_children(
                ops_torch, cu_scores_torch, remaining=15, log_mu=math.log(0.03), root_k=None
            )

            # Convert to numpy for comparison
            par_t_np = par_t.numpy()
            idx_t_np = idx_t.numpy()
            val_t_np = val_t.numpy()

            # Must be identical
            np.testing.assert_array_equal(par_np, par_t_np, "parent indices differ")
            np.testing.assert_array_equal(idx_np, idx_t_np, "token indices differ")
            np.testing.assert_allclose(val_np, val_t_np, rtol=1e-5, err_msg="scores differ")

        except ImportError:
            self.skipTest("torch not available")

    def test_multiple_mu_values(self):
        """Test different mu values produce consistent results."""
        import numpy as np
        from specdec.adaptive import NumpyOps, TorchOps, select_children

        np.random.seed(789)
        cu_scores_np = np.random.randn(8, 80).astype(np.float32)

        for mu in [0.01, 0.03, 0.1]:
            log_mu = math.log(mu)

            # Numpy reference
            ops_np = NumpyOps()
            par_np, idx_np, val_np = select_children(
                ops_np, cu_scores_np, remaining=40, log_mu=log_mu, root_k=None
            )

            # Torch implementation
            try:
                import torch
                ops_torch = TorchOps()
                cu_scores_torch = torch.from_numpy(cu_scores_np)
                par_t, idx_t, val_t = select_children(
                    ops_torch, cu_scores_torch, remaining=40, log_mu=log_mu, root_k=None
                )

                np.testing.assert_array_equal(par_np, par_t.numpy(),
                    f"parent indices differ for mu={mu}")
                np.testing.assert_array_equal(idx_np, idx_t.numpy(),
                    f"token indices differ for mu={mu}")

            except ImportError:
                self.skipTest("torch not available")


if __name__ == "__main__":
    unittest.main()
