# Speculative Decoding {#sec:spec-decode}

Speculative decoding
[@leviathan2023fast; @chen2023accelerating; @stern2018blockwise] is an
inference technique in which a fast, small LM *drafts* candidate tokens
that a larger target LM *verifies* in a single forward pass. By
relieving the target LM of generating every token autoregressively, the
method accelerates inference without altering the output distribution.

Speculative decoding has recently evolved beyond single-model
draft--target pairs. A notable direction is tree-based speculation
[@Miao_2024; @li2024eagle; @li2024eagle2; @li2025eagle3; @chen2024sequoia; @wang2024opttree]:
instead of producing a single linear sequence of draft tokens, the draft
model produces alternative candidate continuations that form a tree.

# Challenges and Key Idea

The open question for tree-based speculation is how to determine the
optimal tree structure that maximizes accepted path length without
incurring excessive verification overhead. Under a fixed candidate token
budget, a deep, narrow tree commits to a few long sequences and
maximizes tokens obtainable per target step, but an early misprediction
invalidates the remaining path. A shallow, wide tree explores more
branches near the root and is more robust, but caps the progress
achievable in a single step.

Prior work treats this choice as fixed. @wang2024opttree adapt tree
structure to maximize expected acceptance using the drafter's own
confidence, and @li2024eagle2 expand and rerank candidates dynamically
within a drafting step; both select *which tokens* populate a tree of a
given size. SGLang [@zheng2024sglang; @sglang_adaptive_spec] adapts
drafting *depth* at runtime from a moving average of accepted length,
but its adaptive mode applies only when branching is disabled. None of
these ask how a fixed budget should be *allocated* between depth and
width, or whether that allocation should depend on the workload being
generated.

Therefore, our primary research question is: *Under a fixed candidate
token budget, can the draft tree topology be dynamically selected based
on workload characteristics?* Our intuition hinges on the structural
properties of different LM tasks:

- **Low-entropy tasks** (e.g. coding): generation is constrained by
  syntax and boilerplate, so token uncertainty is low and the drafter
  stays accurate several steps ahead. We expect these to favor deep,
  narrow trees.

- **High-entropy tasks** (e.g. open-ended writing): prose branches early
  into many semantically valid continuations. We expect these to favor
  shallow, wide trees.

#### Contributions

\(i\) A matched-budget comparison of step-wise acceptance across
drafters, which published figures do not permit since they use differing
tree budgets [@liu2026speculative].

\(ii\) A characterization of how the optimal depth/width allocation
varies with workload entropy and serving batch size.

\(iii\) A training-free heuristic that selects tree shape from task type
and batch size, validated against the best fixed configuration.

#### Scope

We work entirely at inference time on publicly released checkpoints,
training no draft models and proposing no new drafter architecture.

# Data and Approach

#### Models

We use `Llama-3.1-8B-Instruct` as the target, with the publicly released
`EAGLE-LLaMA3.1-Instruct-8B` and `EAGLE3-LLaMA3.1-Instruct-8B` drafters.
This pair exhibits the step-profile divergence central to our
hypothesis: EAGLE's acceptance decays rapidly with draft depth, while
EAGLE-3's remains comparatively flat. We additionally evaluate PRISM
where its released artifact [@prism-github] permits.

#### Workloads

We span the entropy range using coding
[@chen2021humaneval; @austin2021program], math [@cobbe2021gsm8k],
open-ended writing
[@zheng2023mtbench; @taori2023alpaca; @see2017pointer], and question
answering [@kwiatkowski2019nq].

#### Sweep

For each fixed candidate budget $B \in \{16, 32, 60\}$, we enumerate
depth--width pairs $(d, k)$ whose verified node count approximates $B$,
ranging from a linear chain to a shallow fan-out. Because speculative
decoding's benefit depends on spare compute, we repeat the sweep at
batch sizes 1, 8, and 32; published speedups are batch-size-1 figures
and overstate the gain on loaded servers.

Fixing the candidate budget reduces the search to the handful of $(d,k)$
pairs satisfying it, rather than an unconstrained topology space. Two
drafters, four workload categories, three budgets and three batch sizes
yield roughly 200 inference-only runs, which we estimate at under 30
GPU-hours.

#### Metrics

Mean accepted tokens per verification step is our primary measure, as it
is hardware-independent [@li2024eagle; @Miao_2024; @xia2024unlocking].
We also report per-step acceptance rate, end-to-end throughput with
batch size stated, and the runtime split between drafting and
verification as well as the speedup ratio relative to vanilla
autoregressive decoding, following the definition used by @li2025eagle3.
Where acceptance and per-pass cost trade off, end-to-end throughput at a
given batch size is the deciding metric; acceptance length serves to
explain the result rather than determine it.

#### Baselines and success criterion

We compare workload-conditioned selection against the published default
configuration, the single best fixed shape averaged over all workloads,
and SGLang's depth-only adaptive mode. The hypothesis is supported if
the conditioned selection exceeds the best fixed shape by a margin
larger than run-to-run variance across three repeats.

# Literature Review {#sec:related}

**SpecInfer** [@Miao_2024] is the first work to replace linear sequence
drafting with tree-based drafting. The target model verifies candidate
branches in parallel using tree attention; thus, a mispredicted top
candidate does not invalidate the step, since alternative sibling paths
may still be accepted.

**EAGLE** [@li2024eagle] pioneered feature-level drafting by
autoregressively predicting upper-layer hidden states. While EAGLE
exhibits a rapid decay in acceptance rate by drafting depth, EAGLE-3
fuses multi-level feature representations from the target model,
producing an acceptance curve that remains stable across speculative
steps.

**PRISM** [@prism2026] sweeps tree topologies for its own drafter and
reports that deeper, narrower trees outperform shallower, wider ones at
a fixed verified-node count. It treats this as a single global
hyperparameter setting, and notes without testing that the optimum
should depend on the drafter's predictive characteristics. We take that
observation as our starting point, asking whether the optimum also
varies by workload and batch size.

**OPT-Tree** [@wang2024opttree] is closest to our setting, constructing
adaptive draft trees that maximize expected acceptance length from the
drafter's predicted distribution. Its adaptation is per-step and
confidence-driven; it does not condition on workload type, nor examine
how the depth/width trade-off shifts under batching, which is where
speculative decoding's advantage erodes.

# Member Roles {#member-roles .unnumbered}

**Nabanita Bag** identified key research gaps, evaluated the feasibility
of candidate models and benchmarks, and authored the initial draft of
the proposal. **Hemanjali Gasada** introduced the foundational concepts
of speculative decoding to the team and established the technical
background context. **Junyeol Ryu** led the initial brainstorming,
streamlined the project to a high-impact and realistic two-month
timeline, and finalized the proposal draft. Generative AI was used in
part to format this draft and summarize related literature.

[^1]: Overleaf project:
