# Adaptive tree (TALON) vs EAGLE-3: results so far

All runs use Llama-3.1-8B-Instruct as the target and the official EAGLE-3 drafter
(`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`), batch size 1, greedy decoding (T=0), up to 1,024 new
tokens per turn and 80 questions per benchmark (MT-Bench has two turns per question). The Path B folders
below are each one run of `scripts/run_adaptive_suite.sh` (cell 7 of the Colab notebook, or the CHTC job):

- **EAGLE-3:** fixed tree, K=10 children per node, depth 8 (`steps=9`), at budgets N = 32, 48, 60, 96.
- **TALON (adaptive tree):** the same budgets with mu=0.03, plus mu = 0.01, 0.02, 0.04, 0.05 at N=60.
- **Plain decoding:** the speedup baseline, on the same model load.

| Folder | GPU | Benchmarks |
| --- | --- | --- |
| `adaptive_three_bench/` | Colab A100 | Full suite: GSM8K, Sum, QA. Main comparison only (N=60): HumanEval, Alpaca |
| `chtc_h200/adaptive_three_bench/` | CHTC H200 | Full suite: HumanEval, Alpaca, MT-Bench |
| `chtc_l40s/adaptive_gsm8k_sum_qa/` | CHTC L40S | Full suite: GSM8K, Sum, QA |
| `sglang_h200/` | CHTC H200 | Path A (SGLang): 21 shapes on all six benchmarks at batch sizes 1, 8, 32 |
| `sglang_talon_bs8_h200/` | CHTC H200 | Path A: plain decoding, EAGLE-3 and TALON-S at batch 8, all six benchmarks |
| `sglang_talon_bs8_A100/` | Local A100-40GB | Path A: the same batch-8 comparison, repeated on A100 (see `SUMMARY.md` there) |

Each folder has a `summary.csv` (one row per benchmark and configuration) and the per-answer
`.jsonl` files.

**How to read the numbers:**
- **Speedup:** throughput divided by plain decoding's throughput, on the same GPU.
- **τ:** tokens accepted per target-model step.
- **Depth:** the mean depth of the draft tree.

τ and depth depend only on the models and the data, and they agree across GPUs to about ±0.1, so
they can be pooled. Speedups depend on the GPU, so compare them only within one folder.

## Findings

### 1. TALON's tree shape follows how predictable the workload is

With the same budget (N=60), TALON's trees are deep on predictable text and shallow on
unpredictable text. EAGLE-3's fixed tree stays at depth 8.2 to 8.9 on every benchmark.

| Benchmark | EAGLE-3 τ (predictability) | TALON depth | EAGLE-3 depth |
| --- | --- | --- | --- |
| HumanEval (code) | 7.05 | 8.5 | 8.9 |
| MT-Bench (dialogue) | 6.48 | 6.9 | 8.6 |
| GSM8K (math) | 6.44 | 7.0 | 8.8 |
| Alpaca (instructions) | 6.92 | 6.4 | 8.7 |
| Sum (summarisation) | 5.46 | 6.0 | 8.4 |
| QA | 5.59 | 5.8 | 8.2 |

This is the pattern the proposal predicts: deep and narrow trees for low-entropy tasks such as code,
shallow and wide trees for high-entropy tasks. Alpaca is the exception: it is fairly predictable
(τ 6.92) but TALON still builds shallow trees for it.

### 2. Shape adapts, but speed mostly doesn't improve

At N=60, TALON's speedup is within about 3% of EAGLE-3's on HumanEval, Alpaca, GSM8K and MT-Bench
(the exception is GSM8K on the L40S, below). On the predictable benchmarks, EAGLE-3's deep fixed tree is already close to the right
shape. TALON accepts about as many tokens per step, but it costs a little more per step to build its
tree.

TALON does better only on the two least predictable benchmarks, and only on the A100. On QA it gets
3.84× vs 3.66×, and on Sum 3.73× vs 3.56×: about 5% better at N=60 and up to 10% at N=48. On the
L40S the same two benchmarks are a tie (QA 3.11× vs 3.06×, Sum 2.84× vs 2.85×), and TALON is about
4% slower on GSM8K. The tree shapes and τ are the same on both GPUs, so the difference is purely
cost: whether drafting fewer, wider layers saves time depends on how expensive draft passes are
relative to verification on that hardware.

The paper reports that TALON consistently outperforms EAGLE-3. These runs reproduce that only for
low-acceptance workloads on the A100.

### 3. Fewer draft passes, lower acceptance

On unpredictable text, TALON's shallow trees accept fewer tokens per step than EAGLE-3's deep ones.
On Alpaca at N=60, for example, τ is 6.42 vs 6.92. They also need far fewer draft passes (about
6.6 vs 9.3 per step). The two effects roughly cancel. So "prose wants shallow trees" is true only once
drafting cost is counted, not in terms of acceptance alone.

### 4. Budget N

On GSM8K, Sum and QA (A100 and L40S), TALON is fastest at N=60 and slows at N=96, where building a
larger adaptive tree costs more than the extra accepted tokens save (e.g. GSM8K on the A100: 4.23× at
N=60, 4.16× at N=96). EAGLE-3 keeps improving up to N=96 on the A100. On the H200, both methods
change by only a few percent between N=48 and N=96. TALON's relative advantage, where it has one, is
largest at small and medium budgets.

### 5. Threshold mu

Raising mu from 0.01 to 0.05 makes TALON's trees deeper and narrower on every benchmark (e.g. GSM8K
depth goes from 5.5 to 7.9) and raises τ. Throughput barely changes: mu = 0.01 is usually a little slower, values from 0.02
to 0.05 are within a few percent of each other, and the paper's default of 0.03 is within about 3.5% of the best. Tuning mu per
workload would gain little.

## Path A: SGLang at batch sizes 1, 8 and 32

`sglang_h200/` holds the matched-budget sweep from the proposal, run in SGLang v0.5.22 (official
container) on one H200 with SGLang's EAGLE-3 drafter (`lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B`).
It has 21 server configurations on all six benchmarks at batch sizes 1, 8 and 32, which makes 378
runs with no failed requests:

- **Plain decoding:** the speedup baseline.
- **SGLang's own modes:** auto (5 steps × 4 children, budget 8) and adaptive (chain only).
- **Three chains:** budgets 4, 8 and 16.
- **Matched-budget trees:** five depth × width splits at each of budgets 16, 32 and 60, plus the
  EAGLE repo's default tree and the TALON paper's EAGLE-3 tree (9 × 10, budget 60).

Plain decoding gets about 22× faster from batch 1 to batch 32 (208, 1,439 and 4,581 tok/s on average).

### 6. Batch size decides the shape

Speedup over plain decoding, geometric mean of the six workloads:

| Shape | Budget | Batch 1 | Batch 8 | Batch 32 |
| --- | --- | --- | --- | --- |
| Chain, 3 deep | 4 | 2.35× | 2.20× | **1.81×** |
| Chain, 7 deep | 8 | **2.68×** | **2.49×** | 1.74× |
| Tree 7 × 2 | 16 | 2.46× | 2.07× | 1.16× |
| Tree 6 × 3 | 32 | 2.49× | 1.84× | 0.83× |
| Tree 9 × 10 (TALON's EAGLE-3) | 60 | 2.67× | 1.45× | 0.58× |
| SGLang adaptive | chain | 2.48× | 2.22× | 1.54× |
| SGLang auto | 8 | 2.45× | 2.09× | 1.02× |

At batch 1, large trees and a 7-deep chain are about equally good. As the batch grows, the GPU has
no spare compute, so every extra draft token costs real time. At batch 32, every budget-32 and
budget-60 tree is *slower* than plain decoding (0.5–0.9×), and the best shape is the 3-token chain.
Within each budget the best shape also moves towards deeper and narrower as the batch grows: at
batch 32 and budget 16, the chain beats every branching tree on all six workloads.

### 7. Within a budget, the workload effect is real but small

At batch 1 the proposal's hypothesis holds: for the same budget, the fastest split moves from deep
to wide as the workload gets less predictable. At budget 32, HumanEval is fastest with 9 × 2;
GSM8K, MT-Bench and Alpaca with 6 × 3; and Sum and QA with 5 × 4. Budgets 16 and 60 show the same
shift. The deepest tree has the highest τ on every workload. Wider trees win on Sum and QA because
the extra depth is rarely accepted there.

Choosing the shape per workload gains at most about 4% over the best single shape at that budget
(QA: 2.30× vs 2.22×). Across all shapes, a per-workload choice beats the best single fixed shape
by 1.7% at batch 1, 0% at batch 8 and 0.4% at batch 32. Matching the shape to the batch size is
worth up to 3.4× (at batch 32: 1.81× for the 3-token chain vs 0.53× for the EAGLE repo's default
tree). A shape-selection rule should therefore key on batch size first and workload second.

### 8. TALON-S at batch 8: TALON's rules alone do not help

`sglang_talon_bs8_h200/` compares, on one H200 at batch 8 across all six benchmarks, plain decoding,
EAGLE-3 and TALON-S on the same 9 × 10 tree with budget 60. TALON-S
(`specdec/sglang_talon_plugin.py`) applies TALON's tree rules inside SGLang's drafter:
- every one of the 10 root children is kept;
- a later node is kept only if p ≥ 0.03 × the best candidate of its layer and its parent was kept;
- only kept nodes are expanded;
- the budget is filled layer by layer.

Both servers draft identically (9 passes) and verify 60 tokens, so only the choice of which 60
differs. The hook's marker file confirms it ran in SGLang's scheduler process during CUDA-graph
capture.

| Benchmark | EAGLE-3 | TALON-S | τ EAGLE-3 → TALON-S |
| --- | --- | --- | --- |
| HumanEval | 1.62× | 1.52× | 7.04 → 6.74 |
| GSM8K | 1.48× | 1.37× | 6.43 → 6.07 |
| MT-Bench | 1.44× | 1.40× | 6.43 → 6.09 |
| Alpaca | 1.59× | 1.45× | 6.88 → 6.39 |
| Sum | 1.19× | 1.12× | 5.47 → 5.17 |
| QA | 1.27× | 1.20× | 5.54 → 5.23 |
| Geometric mean | 1.42× | 1.33× | |

TALON-S is 3–8% slower on every workload and accepts 4–7% fewer tokens per step. Within a
fixed tree, forcing all root children and filling the budget layer by layer spends tokens on
shallow branches, while EAGLE-2's global top-59 keeps more deep nodes that get accepted. TALON's
saving in Path B came from fewer draft passes on uncertain text, and SGLang's fixed-shape CUDA
graph always runs all 9. The EAGLE-3 row reproduces the sweep in `sglang_h200/` (1.42× vs 1.45×
geometric mean). Token-for-token agreement with plain decoding is the same for both methods:
73–94% for TALON-S and 74–96% for EAGLE-3. MT-Bench is lowest for both, probably because a
first-turn difference carries into the second turn.

**Repeated on A100.** `sglang_talon_bs8_A100/` runs the same config on a 40 GB A100, with
`mem_fraction_static` lowered from 0.80 to 0.70 for both speculative servers (at 0.80 the 9 × 10
tree's CUDA graphs run out of memory at server start; the setting only shrinks the KV cache). Plain
decoding and EAGLE-3 ran on one GPU and TALON-S on a second, identical one. The result is the same:

| Benchmark | EAGLE-3 | TALON-S | τ EAGLE-3 → TALON-S |
| --- | --- | --- | --- |
| HumanEval | 1.49× | 1.39× | 7.04 → 6.75 |
| GSM8K | 1.35× | 1.25× | 6.43 → 6.10 |
| MT-Bench | 1.39× | 1.27× | 6.47 → 6.09 |
| Alpaca | 1.46× | 1.32× | 6.89 → 6.36 |
| Sum | 1.33× | 1.24× | 5.48 → 5.18 |
| QA | 1.17× | 1.09× | 5.56 → 5.25 |
| Geometric mean | 1.36× | 1.26× | |

τ matches the H200 run to within 0.05 on every benchmark, as expected, since τ does not depend on the
GPU. TALON-S is 6–9% slower than EAGLE-3 on the A100. Splitting that gap, about three quarters comes
from the lower τ and the rest from the hook's own cost, which adds 1–2% (about 1 ms) to each 65 ms
batch-8 step. Agreement with plain decoding is again similar for both methods (72–95% for EAGLE-3,
72–93% for TALON-S).

## Correctness

At T=0, speculative decoding should reproduce plain decoding exactly. Comparing token IDs on the A100
HumanEval and Alpaca runs, 89 to 92% of answers are identical under both EAGLE-3 and TALON. The
answers that differ mostly diverge hundreds of tokens in, and EAGLE-3 and TALON often diverge on the
same questions at the same token. That points to fp16 near-ties in the target model rather than a
bug in either tree builder. `scripts/diagnose_mismatch.py` classifies each mismatch and has not yet
been run on these files.

## Caveats

- Path B is batch size 1 only. Path A covers batch sizes 8 and 32 with fixed shapes and SGLang's own drafter; TALON's rules (TALON-S) were run only at batch 8 and only within a fixed tree.
- Path A ran SGLang with `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`, since its EAGLE-3 drafter declares a 2,048-token context. Whether any Sum prompt plus answer goes past 2,048 tokens has not been checked yet.
- 80 questions per benchmark. Speedup confidence intervals in `summary.csv` compare each method with
  plain decoding, not TALON with EAGLE-3 directly. A paired bootstrap would be needed to call the
  A100 QA/Sum gains significant.
- Plain decoding in the EAGLE repo's code is slow (about 19 tok/s on the A100, 44 on the H200), so
  absolute speedups are higher than with an optimised serving engine. Comparisons between methods
  are unaffected.
- On the A100, part of the HumanEval run was split across two Colab sessions. The resume logic keeps
  a checkpoint only if the GPU and settings match.
