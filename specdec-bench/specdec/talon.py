"""TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees.

This module provides the full TALON implementation (Liu et al., arXiv 2601.07353,
Algorithm 2) for the official EAGLE repository (Path B, batch size 1 only).

The core algorithm is implemented in adaptive.py. This module re-exports those
components under the canonical TALON naming:

  - TalonConfig: Configuration for TALON (budget N, threshold mu, root width K)
  - TalonDrafter: Drop-in replacement for EAGLE-3's topK_genrate
  - enable_talon: Monkey-patch an EAGLE model to use TALON drafting

For SGLang (Path A, batch sizes 1/8/32), use specdec.talon_sglang instead.

Example usage (Path B):
    from specdec.talon import TalonConfig, enable_talon

    config = TalonConfig(budget=60, mu=0.03, root_k=10)
    model = load_eagle_model(...)
    enable_talon(model, config)
    # Now model.ea_layer.topK_genrate uses TALON's adaptive tree builder
"""
from .adaptive import (
    AdaptiveConfig as TalonConfig,
    AdaptiveDrafter as TalonDrafter,
    enable_adaptive as enable_talon,
    select_children,
    NumpyOps,
    TorchOps,
)

__all__ = [
    "TalonConfig",
    "TalonDrafter",
    "enable_talon",
    "select_children",
    "NumpyOps",
    "TorchOps",
]
