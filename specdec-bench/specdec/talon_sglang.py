"""TALON node selection for SGLang EAGLE-3 drafter.

This module implements TALON's confidence-gated node selection rule inside SGLang's
EAGLE-3 tree builder. It replaces SGLang's `select_top_k_tokens` with TALON's
Algorithm 2 selection:

  Layer 0 (robust init): Force top-K children of the root (K = root_k)
  Layer d >= 1 (confidence gating): Keep every child whose cumulative probability
    is at least mu * m_d, where m_d is the best candidate in the layer.

Known limitation (TALON-S):
  SGLang uses fixed tensor shapes, so TALON is capped at topk nodes per layer and
  `steps` layers. When a cap binds, free slots are filled by EAGLE-2's rule.
  This variant is called "TALON-S" in reports. Full TALON claims come only from
  Path B (official EAGLE repo).

Usage:
  The plugin is registered via pyproject.toml entry point. To enable:
    export SPECDEC_TALON=1
    export SPECDEC_TALON_BUDGET=60
    export SPECDEC_TALON_MU=0.03
    export SPECDEC_TALON_ROOT_K=10

  Or programmatically:
    from specdec.talon_sglang import TalonSGLangConfig, enable_talon
    config = TalonSGLangConfig(budget=60, mu=0.03, root_k=10, num_steps=9)
    enable_talon(config)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TalonSGLangConfig:
    """Configuration for TALON inside SGLang."""
    budget: int = 60        # N: verified nodes INCLUDING the root
    mu: float = 0.03        # confidence threshold, 0 < mu <= 1
    root_k: int = 10        # K: width of the forced top-K root layer
    num_steps: int = 9      # SGLang tree depth (steps = EAGLE depth + 1)

    @classmethod
    def from_env(cls) -> Optional["TalonSGLangConfig"]:
        """Load config from environment variables if SPECDEC_TALON=1."""
        if os.environ.get("SPECDEC_TALON", "0") != "1":
            return None
        return cls(
            budget=int(os.environ.get("SPECDEC_TALON_BUDGET", "60")),
            mu=float(os.environ.get("SPECDEC_TALON_MU", "0.03")),
            root_k=int(os.environ.get("SPECDEC_TALON_ROOT_K", "10")),
            num_steps=int(os.environ.get("SPECDEC_TALON_NUM_STEPS", "9")),
        )

    def validate(self) -> "TalonSGLangConfig":
        if self.budget < 2:
            raise ValueError("budget must be >= 2")
        if not (0.0 < self.mu <= 1.0):
            raise ValueError("mu must be in (0, 1]")
        if self.root_k < 1:
            raise ValueError("root_k must be >= 1")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        return self

    def to_dict(self):
        return {
            "budget": self.budget,
            "mu": self.mu,
            "root_k": self.root_k,
            "num_steps": self.num_steps,
        }


def talon_select_tokens(
    logprobs,           # (batch, topk, vocab) or (batch, vocab)
    parent_scores,      # (batch, topk) cumulative log-probs of parents
    layer: int,
    config: TalonSGLangConfig,
    topk: int,
    remaining_budget: int,
):
    """TALON's confidence-gated token selection for one layer.

    Args:
        logprobs: Log-probabilities from the draft model
        parent_scores: Cumulative log-probabilities of parent nodes
        layer: Current tree layer (0 = first layer after root)
        config: TALON configuration
        topk: SGLang's topk parameter (max nodes per layer)
        remaining_budget: Nodes remaining in the budget

    Returns:
        selected_indices: Indices of selected nodes
        selected_scores: Cumulative scores of selected nodes
        n_selected: Number of nodes selected by TALON (rest filled by EAGLE-2 rule)
    """
    import torch

    batch_size = logprobs.shape[0]

    if layer == 0:
        # Layer 0: Robust top-K initialization
        # Select top-K children of the root
        k = min(config.root_k, remaining_budget - 1, logprobs.shape[-1])

        if logprobs.dim() == 2:
            # (batch, vocab) -> select top-k
            values, indices = torch.topk(logprobs, k, dim=-1)
            scores = values  # root score is 0 in log space
        else:
            # (batch, 1, vocab) for single parent
            values, indices = torch.topk(logprobs[:, 0], k, dim=-1)
            scores = values

        return indices, scores, k

    else:
        # Layer d >= 1: Confidence-gated expansion
        # S_d = all children of current parents, scored by cumulative probability
        # Keep u if p(u) >= mu * m_d where m_d = max_{u in S_d} p(u)

        log_mu = math.log(config.mu)

        if logprobs.dim() == 2:
            # Single parent case
            cu_scores = parent_scores[:, :1] + logprobs  # (batch, vocab)
        else:
            # Multiple parents: (batch, n_parents, vocab)
            n_parents = logprobs.shape[1]
            cu_scores = parent_scores[:, :n_parents, None] + logprobs  # (batch, n_par, vocab)

        # Find anchor (best score in the layer)
        if cu_scores.dim() == 2:
            anchor = cu_scores.max(dim=-1, keepdim=True).values  # (batch, 1)
            passing_mask = cu_scores >= anchor + log_mu
        else:
            flat_scores = cu_scores.reshape(batch_size, -1)  # (batch, n_par * vocab)
            anchor = flat_scores.max(dim=-1, keepdim=True).values  # (batch, 1)
            passing_mask = flat_scores >= anchor + log_mu  # (batch, n_par * vocab)
            cu_scores = flat_scores

        # Count passing candidates
        n_passing = passing_mask.sum(dim=-1)  # (batch,)
        max_passing = n_passing.max().item()

        # Budget check: keep at most remaining_budget nodes
        k = min(max_passing, remaining_budget, topk)

        if k <= 0:
            k = 1  # Always keep at least the anchor

        # Select top-k among passing candidates (or just top-k if none pass)
        scores_filtered = torch.where(passing_mask, cu_scores, torch.tensor(float('-inf')))
        selected_scores, selected_indices = torch.topk(scores_filtered, k, dim=-1)

        # Count how many were actually selected by TALON (not filler)
        n_talon = (selected_scores > float('-inf')).sum(dim=-1)

        return selected_indices, selected_scores, n_talon.min().item()


# Global state for the active TALON configuration
_active_config: Optional[TalonSGLangConfig] = None


def enable_talon(config: Optional[TalonSGLangConfig] = None):
    """Enable TALON node selection in SGLang.

    If config is None, loads from environment variables.
    """
    global _active_config
    if config is None:
        config = TalonSGLangConfig.from_env()
    if config:
        config.validate()
    _active_config = config


def get_talon_config() -> Optional[TalonSGLangConfig]:
    """Get the active TALON configuration."""
    return _active_config


def is_talon_enabled() -> bool:
    """Check if TALON is enabled."""
    return _active_config is not None
