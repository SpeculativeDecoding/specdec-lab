"""Draft-tree configuration shared by both code paths.

Verified against source (Oct 2026):

SGLang v0.5.22 (python/sglang/srt/speculative/eagle_worker_v2.py, eagle_utils.py)
  --speculative-num-steps S          number of draft layers (tree depth)
  --speculative-eagle-topk K         children per parent, parents kept per layer
  --speculative-num-draft-tokens B   verified nodes INCLUDING the root token
  candidate pool = K + (S - 1) * K^2  (organize_draft_results: torch.topk(score_list, B - 1))
  if K == 1, SGLang forces B = S + 1  (arg_groups/speculative_hook.py, _handle_eagle_family)

EAGLE repo @ cb7e084 (eagle/model/cnets.py: topK_genrate, ea_model.py)
  total_token = B   (help text: "number of drafted tokens in the tree + 1";
                     ea_layer.total_tokens = total_token - 1 draft nodes are kept)
  depth       = S - 1 (one draft pass before the loop, then `depth` more)
  top_k       = K
  candidate pool = top_k + depth * top_k^2  -> identical to SGLang's.

So (S, K, B) in SGLang == (depth=S-1, top_k=K, total_token=B) in the EAGLE repo.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TreeConfig:
    steps: int             # SGLang --speculative-num-steps (tree depth in layers)
    topk: int              # SGLang --speculative-eagle-topk
    num_draft_tokens: int  # SGLang --speculative-num-draft-tokens (budget B, incl. root)
    label: str = ""

    # ---- derived quantities -------------------------------------------------
    @property
    def pool(self) -> int:
        """Number of drafted candidate nodes the verifier can choose from."""
        return self.topk + (self.steps - 1) * self.topk * self.topk

    @property
    def eagle_depth(self) -> int:
        return self.steps - 1

    @property
    def eagle_total_token(self) -> int:
        return self.num_draft_tokens

    @property
    def eagle_top_k(self) -> int:
        return self.topk

    @property
    def name(self) -> str:
        base = f"s{self.steps}_k{self.topk}_b{self.num_draft_tokens}"
        return f"{base}_{self.label}" if self.label else base

    # ---- validation ---------------------------------------------------------
    def problems(self) -> List[str]:
        out = []
        if self.steps < 1:
            out.append("steps must be >= 1")
        if self.topk < 1:
            out.append("topk must be >= 1")
        if self.num_draft_tokens < 2:
            out.append("num_draft_tokens must be >= 2 (root + at least one draft)")
        if self.topk == 1 and self.num_draft_tokens != self.steps + 1:
            out.append(
                f"topk=1 requires num_draft_tokens = steps + 1 = {self.steps + 1} "
                "(SGLang silently overrides it otherwise)"
            )
        if self.num_draft_tokens - 1 > self.pool:
            out.append(
                f"budget needs {self.num_draft_tokens - 1} draft nodes but the pool "
                f"only has {self.pool} (= topk + (steps-1)*topk^2); torch.topk would fail"
            )
        return out

    def validate(self) -> "TreeConfig":
        p = self.problems()
        if p:
            raise ValueError(f"Invalid tree config {self.name}: " + "; ".join(p))
        return self

    # ---- export ---------------------------------------------------------------
    def sglang_args(self) -> List[str]:
        return [
            "--speculative-num-steps", str(self.steps),
            "--speculative-eagle-topk", str(self.topk),
            "--speculative-num-draft-tokens", str(self.num_draft_tokens),
        ]

    def eagle_kwargs(self) -> Dict[str, int]:
        return {
            "total_token": self.eagle_total_token,
            "depth": self.eagle_depth,
            "top_k": self.eagle_top_k,
        }

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.update(pool=self.pool, name=self.name,
                 eagle_depth=self.eagle_depth, eagle_total_token=self.eagle_total_token)
        return d


# Matched-budget grid from the execution plan (SGLang notation: steps, topk, B).
PLAN_GRID: Dict[int, List[TreeConfig]] = {
    16: [
        TreeConfig(15, 1, 16, "chain"),
        TreeConfig(7, 2, 16),
        TreeConfig(5, 3, 16),
        TreeConfig(3, 4, 16),
        TreeConfig(2, 8, 16),
    ],
    32: [
        TreeConfig(9, 2, 32),
        TreeConfig(6, 3, 32),
        TreeConfig(5, 4, 32),
        TreeConfig(4, 6, 32),
        TreeConfig(3, 8, 32),
    ],
    60: [
        TreeConfig(16, 2, 60),
        TreeConfig(10, 3, 60),
        TreeConfig(8, 4, 60),
        TreeConfig(6, 6, 60),
        # == EAGLE repo eval default (total_token=60, depth=5, top_k=10)
        TreeConfig(6, 10, 60, "eagle_repo_default"),
        # TALON paper's EAGLE-3 baseline: K=10, D=8, N=60. Its Algorithm 1 loops d = 0..D,
        # i.e. D + 1 draft layers (delta = D + 1), which is EAGLE-repo depth=8 = steps 9.
        TreeConfig(9, 10, 60, "talon_eagle3_baseline"),
    ],
}

# Reference points outside the matched grid.
SGLANG_AUTO_LLAMA = TreeConfig(5, 4, 8, "sglang_auto")  # _auto_choose_speculative_params
CHAIN_REFERENCES = [TreeConfig(3, 1, 4, "chain"), TreeConfig(7, 1, 8, "chain")]


def plan_grid(budgets=(16, 32, 60)) -> List[TreeConfig]:
    out: List[TreeConfig] = []
    for b in budgets:
        out.extend(PLAN_GRID[b])
    for cfg in out:
        cfg.validate()
    return out


def from_dict(d: Dict) -> TreeConfig:
    return TreeConfig(int(d["steps"]), int(d["topk"]), int(d["num_draft_tokens"]),
                      str(d.get("label", ""))).validate()
