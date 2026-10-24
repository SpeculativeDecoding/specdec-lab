"""SGLang plugin for TALON node selection.

This module is registered as an SGLang plugin via pyproject.toml entry point.
When SGLang loads, it calls the `plugin` function which patches the EAGLE-3
tree builder to use TALON's selection rule instead of static top-K.

The patch is only applied when SPECDEC_TALON=1 is set in the environment.

Plugin registration (pyproject.toml):
  [project.entry-points."sglang.srt.plugins"]
  talon = "specdec.sglang_talon_patch:plugin"
"""
from __future__ import annotations

import json
import os
import logging
from datetime import datetime
from typing import Optional

from .talon_sglang import (
    TalonSGLangConfig,
    enable_talon,
    get_talon_config,
    is_talon_enabled,
    talon_select_tokens,
)

logger = logging.getLogger(__name__)


class TalonPatchState:
    """Tracks the state of TALON patching."""

    def __init__(self):
        self.patched = False
        self.original_select = None
        self.config: Optional[TalonSGLangConfig] = None
        self.layer_counter = 0
        self.tree_counter = 0

    def reset_tree(self):
        """Reset counters for a new tree."""
        self.layer_counter = 0
        self.tree_counter += 1


_state = TalonPatchState()


def _write_talon_marker(server_dir: str, config: TalonSGLangConfig):
    """Write a marker file indicating TALON is active with these settings."""
    marker_path = os.path.join(server_dir, "server.log.talon.jsonl")
    marker = {
        "timestamp": datetime.now().isoformat(),
        "patched": True,
        "budget": config.budget,
        "mu": config.mu,
        "root_k": config.root_k,
        "num_steps": config.num_steps,
    }
    try:
        with open(marker_path, "a") as f:
            f.write(json.dumps(marker) + "\n")
        logger.info(f"TALON marker written to {marker_path}")
    except Exception as e:
        logger.warning(f"Failed to write TALON marker: {e}")


def _patched_select_top_k_tokens(
    logprobs,
    parent_cumulative_scores,
    tree_config,
    *args,
    **kwargs,
):
    """Patched version of select_top_k_tokens that uses TALON selection."""
    config = get_talon_config()

    if config is None:
        # TALON not enabled, fall back to original
        return _state.original_select(
            logprobs, parent_cumulative_scores, tree_config, *args, **kwargs
        )

    # Get SGLang's tree parameters
    topk = tree_config.get("topk", 10)
    num_steps = tree_config.get("num_steps", 9)
    budget = tree_config.get("num_draft_tokens", 60)

    # Calculate remaining budget
    # This is an approximation since we don't track exact tree state
    layer = _state.layer_counter
    remaining = budget - 1  # -1 for root

    try:
        # Call TALON selection
        indices, scores, n_talon = talon_select_tokens(
            logprobs,
            parent_cumulative_scores,
            layer,
            config,
            topk,
            remaining,
        )

        _state.layer_counter += 1

        # If TALON selected fewer than topk, we need to fill with EAGLE-2 rule
        # This is handled inside talon_select_tokens

        return indices, scores

    except Exception as e:
        logger.warning(f"TALON selection failed at layer {layer}, falling back: {e}")
        return _state.original_select(
            logprobs, parent_cumulative_scores, tree_config, *args, **kwargs
        )


def plugin():
    """SGLang plugin entry point.

    Called by SGLang when loading plugins. Patches the EAGLE-3 tree builder
    to use TALON selection when SPECDEC_TALON=1 is set.
    """
    # Check if TALON should be enabled
    if os.environ.get("SPECDEC_TALON", "0") != "1":
        logger.debug("TALON plugin loaded but not enabled (SPECDEC_TALON != 1)")
        return

    # Load configuration from environment
    config = TalonSGLangConfig.from_env()
    if config is None:
        logger.warning("TALON enabled but failed to load config")
        return

    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Invalid TALON config: {e}")
        return

    enable_talon(config)
    _state.config = config

    logger.info(f"TALON plugin enabled: budget={config.budget}, mu={config.mu}, "
                f"root_k={config.root_k}, num_steps={config.num_steps}")

    # Try to patch SGLang's select_top_k_tokens
    try:
        # Import SGLang's eagle_utils module
        from sglang.srt.speculative import eagle_utils

        # Save original function
        if hasattr(eagle_utils, "select_top_k_tokens"):
            _state.original_select = eagle_utils.select_top_k_tokens
            eagle_utils.select_top_k_tokens = _patched_select_top_k_tokens
            _state.patched = True
            logger.info("TALON: Patched eagle_utils.select_top_k_tokens")
        else:
            logger.warning("TALON: select_top_k_tokens not found in eagle_utils")

    except ImportError as e:
        logger.warning(f"TALON: Could not import eagle_utils: {e}")
        logger.warning("TALON patch not applied - running without TALON selection")
    except Exception as e:
        logger.error(f"TALON: Unexpected error during patching: {e}")

    # Write marker file if patched successfully
    if _state.patched:
        # Try to find server output directory
        out_dir = os.environ.get("SPECDEC_TALON_OUT_DIR", ".")
        _write_talon_marker(out_dir, config)


def check_worker(server_spec, out_dir: str) -> bool:
    """Check that a TALON server has the correct marker file.

    Used by the benchmark launcher to verify TALON is actually active.

    Args:
        server_spec: ServerSpec with TALON configuration
        out_dir: Directory where server output is written

    Returns:
        True if the marker file exists and matches the expected config.
    """
    marker_path = os.path.join(out_dir, "server.log.talon.jsonl")

    if not os.path.exists(marker_path):
        logger.warning(f"TALON marker not found: {marker_path}")
        return False

    try:
        with open(marker_path, "r") as f:
            lines = f.readlines()

        if not lines:
            logger.warning("TALON marker file is empty")
            return False

        # Check most recent marker
        marker = json.loads(lines[-1])

        if not marker.get("patched", False):
            logger.warning("TALON marker says patched=false")
            return False

        # Verify config matches
        expected = server_spec.talon_config if hasattr(server_spec, "talon_config") else None
        if expected:
            for key in ["budget", "mu", "root_k", "num_steps"]:
                expected_val = getattr(expected, key, None)
                marker_val = marker.get(key)
                if expected_val is not None and marker_val != expected_val:
                    logger.warning(f"TALON config mismatch: {key} expected {expected_val}, got {marker_val}")
                    return False

        return True

    except Exception as e:
        logger.warning(f"Failed to verify TALON marker: {e}")
        return False


def reset_tree_state():
    """Reset TALON state for a new tree construction."""
    _state.reset_tree()
