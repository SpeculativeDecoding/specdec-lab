"""Speculative-decoding benchmark harness: EAGLE-3 and TALON on Llama-3.1-8B-Instruct.

TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees
(Liu et al., arXiv 2601.07353)

Path A: SGLang (all batch sizes, TALON-S variant).
Path B: Official EAGLE repository (batch size 1, full TALON).
"""
__version__ = "0.1.0"

# TALON exports (Path B: full TALON for EAGLE repo)
from .talon import TalonConfig, TalonDrafter, enable_talon

# Also export under "adaptive" names for backward compatibility
from .adaptive import AdaptiveConfig, AdaptiveDrafter, enable_adaptive

# TALON-S exports (Path A: SGLang)
from .talon_sglang import TalonSGLangConfig, enable_talon as enable_talon_sglang
