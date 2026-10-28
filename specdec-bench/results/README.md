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
| `sglang_talon_bs1_32_h200/` | CHTC H200 | Path A: plain decoding, EAGLE-3 and TALON-S at batch 1 and 32, all six benchmarks |
| `sglang_bscurve_humaneval_alpaca_h200/` | CHTC H200 | Path A: 10 servers (chains, one tree per budget, EAGLE-3, TALON-S, SGLang adaptive) at batch 2, 4, 16 |
| `depth_grid_humaneval_gsm8k_mt_bench_h200/`, `depth_grid_alpaca_sum_qa_h200/` | CHTC H200 | Path B: EAGLE-3 8 × 10, TALON, and fixed depth 1, 2, 4, 6, 12, 20 at width 10; all six benchmarks |
| `dxw80_sweep_{humaneval,gsm8k,sum,qa}_h200/` | CHTC H200 | Path B: depth × width = 80, from 1 × 80 to 40 × 2 (one warm-up per run) |
| `width_grid_humaneval_gsm8k_mt_bench_h200/` | CHTC H200 | Path B: 12 × 4, 20 × 2, 20 × 4, 1 × 20, 2 × 20, 4 × 16 |
| `summary_vs_eagle3.csv` | CHTC H200 | The headline table below (`scripts/compare_vs_eagle3.py`) |
| `talon_depthcap_v2_{gsm8k,humaneval,qa,sum}_h200/` | CHTC H200 | Path B: TALON capped at EAGLE-3's best depth, with and without filling the budget, and with root width 16 (`scripts/run_talon_depthcap.sh`) |
| `talon_depthcap_{gsm8k,humaneval,qa,sum}_h200/` | CHTC H200 | First depth-cap runs, cap one layer too short (EAGLE-3 and uncapped-TALON rows still valid) |
| `template_sweep_{humaneval,gsm8k,qa,sum}_h200/` | CHTC H200 | Path B: 24 fixed-shape templates (exact nodes per depth) plus EAGLE-3 8 × 10 and plain decoding (`scripts/run_template_sweep.sh`) |
| `repeat{2,3}_{humaneval_gsm8k_mt_bench,alpaca_sum_qa}_h200/` | CHTC H200 | Path B: second and third runs of EAGLE-3 8 × 10, TALON, 6 × 10 and 5 × 16 (`scripts/run_repeat.sh`) |
| `diagnose_mismatch_h200/` | CHTC H200 | Every answer that differs from plain decoding, classified (`scripts/run_diagnose.sh`) |
| `talon_shapes_alpaca_sum_qa_h200/`, `talon_shapes_humaneval_gsm8k_mt_bench_a100/` | CHTC H200 / A100-80GB | Path B: TALON with every tree's shape recorded, and `shapes.csv` (section 10). The second job landed on an A100, so use its shapes, not its speeds |

Each folder has a `summary.csv` (one row per benchmark and configuration) and the per-answer
`.jsonl` files.

**How to read the numbers:**
- **Plain decoding** (`ar` in file names) is the target model alone with greedy decoding: one token
  per forward pass, no drafter. Every method decodes greedily (T=0), so all of them produce the
  same text, just at different speeds.
- **Speedup:** throughput divided by plain decoding's throughput, on the same GPU.
- **vs EAGLE-3:** throughput divided by EAGLE-3's 8 × 10 tree, measured on the same questions.
- **τ:** tokens accepted per target-model step.
- **Depth:** the mean depth of the draft tree.

τ and depth depend only on the models and the data, and they agree across GPUs to about ±0.1, so
they can be pooled. Speedups depend on the GPU, so compare them only within one folder.

## Headline: EAGLE-3 vs TALON vs "Ours" (Path B, H200, batch 1)

Three ways to spend the same budget of 60 verified tokens per step:

- **EAGLE-3:** one fixed tree, depth 8 × width 10, for every workload.
- **TALON:** reshapes the tree at every step from the drafter's confidence (mu = 0.03, K = 10).
- **Ours:** one fixed tree per workload, chosen offline. For each workload, the fixed tree with the
  best speedup on the calibration questions (frozen 30% split, `specdec.data.split`) among every
  fixed tree run on that workload: up to 21 shapes from the depth grid, the depth × width = 80
  sweep and the width grid, EAGLE-3's own 8 × 10 included.

Everything below is scored on the other 70% (test questions: 51–63 per workload, since the split
hashes question ids), all on H200s.

| Workload | EAGLE-3 vs plain | TALON vs EAGLE-3 | Ours: chosen tree | Ours vs EAGLE-3 |
| --- | --- | --- | --- | --- |
| HumanEval (code) | 4.42× | 0.99 [0.97, 1.01] | 8 × 10 | 1.00 (same tree) |
| GSM8K (math) | 3.93× | 1.01 [1.00, 1.02] | 8 × 10 | 1.00 (same tree) |
| MT-Bench (dialogue) | 4.06× | 1.01 [0.99, 1.02] | 8 × 10 | 1.00 (same tree) |
| Alpaca (instructions) | 4.32× | 0.99 [0.98, 1.01] | 8 × 10 | 1.00 (same tree) |
| Sum (summaries) | 3.41× | **1.04** [1.02, 1.05] | 6 × 10 | **1.04** [1.03, 1.05] |
| QA | 3.32× | **1.07** [1.04, 1.09] | 6 × 10 | **1.05** [1.02, 1.07] |

The bracketed ranges are 95% paired-bootstrap intervals over test questions. A range that
excludes 1.00 means the method is reliably different from EAGLE-3 on these questions.

- **Code, math, dialogue and instructions:** the calibration questions pick EAGLE-3's own 8 × 10, and
  TALON is within ±1% of it. The fixed default is already the right shape for these workloads.
- **Summaries and QA:** a shallower tree wins. Depth 6 × 10 beats EAGLE-3 by 4–5%, which is about
  as much as TALON (4–7%) and is chosen offline with no per-step logic.
- **Extreme shapes never win.** Very shallow and wide trees (1 × 80, 2 × 40) and very deep and
  narrow ones (20 × 4, 40 × 2) lose 15–60% against 8 × 10 (tables below). Deeper trees accept more
  tokens per step but pay one draft pass per layer.

### Acceptance (τ) and drafting cost behind the headline

Same test questions. τ is tokens accepted per verification step of the target model, and "draft
passes" is drafter forward passes per step. Speed depends on both: more accepted tokens per step,
but each draft pass costs time.

| Workload | EAGLE-3 τ / passes | TALON τ / passes | Ours τ / passes |
| --- | --- | --- | --- |
| HumanEval | 7.05 / 9.2 | **7.21** / 8.6 | 7.05 / 9.2 (8 × 10) |
| GSM8K | **6.36** / 9.3 | 6.15 / 7.2 | **6.36** / 9.3 (8 × 10) |
| MT-Bench | **6.52** / 9.2 | 6.33 / 7.1 | **6.52** / 9.2 (8 × 10) |
| Alpaca | **6.98** / 9.3 | 6.46 / 6.7 | **6.98** / 9.3 (8 × 10) |
| Sum | **5.48** / 9.2 | 5.23 / 6.1 | 5.23 / 7.1 (6 × 10) |
| QA | **5.38** / 9.3 | 5.14 / 5.8 | 5.14 / 7.2 (6 × 10) |

- **TALON accepts fewer tokens per step than EAGLE-3 on five of six workloads** (HumanEval is the
  exception) but runs 1.5–3.5 fewer draft passes, because it stops drafting once its budget is
  spent. The two effects cancel on code, math, dialogue and instructions, and favour TALON on
  summaries and QA. Lower τ alone is therefore not a weakness.
- **On Sum and QA, the faster methods accept fewer tokens.** "Ours" (6 × 10) and TALON accept about
  5% less than EAGLE-3 there, yet run 4–7% faster: on unpredictable text EAGLE-3's last two layers
  are rarely accepted, so they cost more than they return.
- **τ alone would pick the wrong tree.** τ rises with depth on every workload and levels off around
  depth 12–20 (table below), yet depth 20 is 25–30% slower than depth 8. Throughput decides; τ
  explains it.

τ by depth (width 10, all questions):

| Depth | HumanEval | GSM8K | MT-Bench | Alpaca | Sum | QA |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.86 | 2.89 | 2.89 | 2.94 | 2.82 | 2.79 |
| 2 | 3.70 | 3.75 | 3.75 | 3.84 | 3.57 | 3.56 |
| 4 | 5.16 | 5.14 | 5.08 | 5.33 | 4.64 | 4.62 |
| 6 | 6.30 | 6.00 | 5.98 | 6.34 | 5.23 | 5.32 |
| 8 (EAGLE-3) | 7.07 | 6.42 | 6.48 | 6.92 | 5.47 | 5.58 |
| 12 | 7.62 | 6.62 | 6.78 | 7.25 | 5.62 | 5.74 |
| 20 | 7.69 | 6.61 | 6.87 | 7.24 | 5.64 | 5.79 |
| TALON | 7.22 | 6.17 | 6.25 | 6.41 | 5.21 | 5.47 |

### Relation to the TALON paper

The paper's main table uses the same setting as ours: one H200, batch 1, T=0, budget 60, mu=0.03 and
the same six benchmarks, with a Llama-3 8B row. Our τ matches that row closely for both methods, so
our EAGLE-3 8 × 10 baseline and our TALON build the same trees as the paper's:

| Workload | Paper τ, EAGLE-3 / TALON | Our τ, EAGLE-3 / TALON | Paper TALON ÷ EAGLE-3 | Ours, runs 1, 2, 3 |
| --- | --- | --- | --- | --- |
| HumanEval | 7.08 / 7.28 | 7.07 / 7.22 | 1.04 | 0.99, 1.02, 1.03 |
| GSM8K | 6.42 / 6.24 | 6.42 / 6.17 | 1.05 | 1.00, 1.02, (1.07) |
| MT-Bench | 6.38 / 6.22 | 6.48 / 6.25 | 1.05 | 1.01, 1.01, (1.08) |
| Alpaca | 6.93 / 6.51 | 6.92 / 6.41 | 1.03 | 0.99, 1.01, 0.99 |
| Sum | 5.46 / 5.28 | 5.47 / 5.21 | 1.08 | 1.04, 1.06, 1.03 |
| QA | 5.36 / 5.09 | 5.58 / 5.47 | 1.09 | 1.08, 1.10, 1.08 |

Our ratios use all 80 questions per workload. Bracketed values come from run 3, whose first
configurations (EAGLE-3 and plain decoding) ran about 7% slower than in every other run (see
Caveats).

We reproduce the paper's gain on QA and Sum but not its 3–5% gain on code, math, dialogue and
instructions. Since the trees match, the difference is in step time: our EAGLE-3 is 9–10% faster
than the paper's relative to plain decoding (4.43× vs 4.05× on HumanEval, at the same τ), while
our TALON is about 5% faster than the paper's. TALON wins by running fewer draft passes (about 6–7
per step instead of 9), so its lead shrinks when the baseline's steps are cheap. The paper does not
name its code base, so this explanation is not confirmed. Our TALON is a reimplementation of
Algorithm 2 (`specdec/adaptive.py`), not the authors' code.

TALON-S (Path A, sections 8 and 9) is our port of TALON's rules into SGLang's fixed-shape drafter.
It cannot skip draft passes, so its slowdown there says nothing about the paper's batch-1 claim.

Regenerate with
`python scripts/compare_vs_eagle3.py results/depth_grid_*_h200 results/dxw80_sweep_*_h200 results/width_grid_*_h200 --out results/summary_vs_eagle3.csv`.

### Every fixed tree relative to EAGLE-3 (all questions)

Depth grid, width 10 (each folder against its own 8 × 10; EAGLE-3's speedup over plain decoding in
the last row):

| Depth | HumanEval | GSM8K | MT-Bench | Alpaca | Sum | QA |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.55 | 0.62 | 0.61 | 0.59 | 0.70 | 0.68 |
| 2 | 0.67 | 0.76 | 0.75 | 0.74 | 0.86 | 0.84 |
| 4 | 0.86 | 0.95 | 0.93 | 0.92 | 1.01 | 0.99 |
| 6 | 0.97 | **1.01** | 0.99 | 1.00 | **1.04** | **1.04** |
| **8 (EAGLE-3)** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 12 | 0.93 | 0.89 | 0.91 | 0.91 | 0.89 | 0.89 |
| 20 | 0.75 | 0.71 | 0.74 | 0.73 | 0.72 | 0.72 |
| TALON | 0.99 | 1.00 | 1.01 | 1.00 | 1.04 | **1.08** |
| EAGLE-3 vs plain | 4.43× | 3.96× | 4.04× | 4.28× | 3.40× | 3.46× |

Depth × width = 80 (each workload against its own 8 × 10):

| Depth × width | HumanEval | GSM8K | Sum | QA |
| --- | --- | --- | --- | --- |
| 1 × 80 | 0.56 | 0.62 | 0.73 | 0.70 |
| 2 × 40 | 0.67 | 0.76 | 0.86 | 0.85 |
| 4 × 20 | 0.88 | 0.88 | 1.02 | 1.00 |
| 5 × 16 | 0.93 | 0.97 | **1.04** | **1.02** |
| **8 × 10** | 1.00 | 1.00 | 1.00 | 1.00 |
| 10 × 8 | 0.97 | 0.94 | 0.96 | 0.95 |
| 16 × 5 | 0.84 | 0.77 | 0.78 | 0.79 |
| 20 × 4 | 0.74 | 0.67 | 0.69 | 0.70 |
| 40 × 2 | 0.46 | 0.39 | 0.40 | 0.41 |

Width grid (ratio of speedups over each run's own plain decoding, against the depth grid's 8 × 10):

| Depth × width | HumanEval | GSM8K | MT-Bench |
| --- | --- | --- | --- |
| 12 × 4 | 0.91 | 0.86 | 0.88 |
| 20 × 2 | 0.66 | 0.59 | 0.60 |
| 20 × 4 | 0.73 | 0.68 | 0.70 |
| 1 × 20 | 0.56 | 0.62 | 0.62 |
| 2 × 20 | 0.66 | 0.74 | 0.74 |
| 4 × 16 | 0.87 | 0.94 | 0.93 |

EAGLE-3's 8 × 10 comes out the same with one warm-up per run (depth × width = 80 sweep) and with
three (depth grid), within 2% on all four shared workloads (HumanEval 4.53× vs 4.43×, GSM8K 3.97×
vs 3.96×, Sum 3.37× vs 3.40×, QA 3.48× vs 3.46×).

### TALON held to EAGLE-3's best depth

`talon_depthcap_v2_*_h200/` caps TALON at the number of layers of EAGLE-3's best fixed tree for
each workload, taken from the depth × width = 80 sweep:
- 8 × 10, 9 layers, for GSM8K and HumanEval;
- 5 × 16, 6 layers, for QA and Sum. EAGLE "depth D" (`--steps D+1`) builds D + 1 layers below the root.

The runs compared are:
- **capped TALON** as is;
- **capped + fill** (`--adaptive-fill`): budget left at the cap goes to the best unused candidates
  of the built layers, so the tree verifies all 60 tokens with no extra draft pass;
- **TALON with root width 16**, EAGLE-3's best width for QA and Sum.

Speed is relative to EAGLE-3's best tree in the same job:

| Workload | TALON | TALON capped | TALON capped + fill | TALON root 16 | Root 16, capped | Root 16, capped + fill |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K | 1.01 | 1.00 | 1.01 | – | – | – |
| HumanEval | 0.99 | 0.99 | 0.98 | – | – | – |
| QA | 1.04 | 0.97 | 0.99 | 1.02 | 0.98 | 0.96 |
| Sum | 0.97 | 0.98 | 0.98 | 1.02 | 0.99 | 0.97 |

Configurations repeated between the first and corrected runs differ by 2–3%, so every entry is
within noise of EAGLE-3's best fixed tree.

- **Capping TALON at EAGLE-3's best depth never helps (0.96–1.00).** Capped trees verify only 50–57
  of 60 tokens. Filling the rest barely changes τ (GSM8K 6.03 → 6.04, QA 4.77 → 4.86), because the
  extra leaves are rarely accepted.
- **TALON's edge, where it has one, comes from depth beyond EAGLE-3's.** Uncapped, it puts about
  6.6 of 60 nodes below depth 9 on HumanEval (down to depth 21), but only 1–2 on GSM8K, QA and Sum.

Mean width at each depth (depth 1 onward; `mean_layer_nodes` in the `.jsonl` files):

| Workload | Tree | Widths by depth |
| --- | --- | --- |
| GSM8K | EAGLE-3 8 × 10 | 3.1, 5.3, 6.8, 7.7, 8.2, 8.0, 7.6, 6.8, 5.6 |
| GSM8K | TALON | 10, 5.4, 7.6, 8.6, 7.7, 6.3, 4.8, 3.3, 2.2, 1.3, then 1.7 deeper |
| HumanEval | EAGLE-3 8 × 10 | 2.8, 4.7, 5.8, 6.6, 7.2, 7.6, 7.9, 8.0, 8.5 |
| HumanEval | TALON | 10, 4.7, 5.9, 6.3, 6.1, 5.3, 4.5, 3.5, 3.2, 2.8, then 6.6 deeper |
| QA | EAGLE-3 5 × 16 | 5.9, 9.7, 11.0, 11.2, 10.8, 10.3 |
| QA | TALON | 10, 9.9, 11.6, 9.5, 6.3, 4.0, 2.5, 1.5, 0.9, 0.5, then 2.3 deeper |
| Sum | EAGLE-3 5 × 16 | 4.9, 9.0, 11.0, 11.6, 11.5, 11.0 |
| Sum | TALON | 10, 8.6, 11.2, 9.3, 6.8, 4.7, 3.0, 1.9, 1.3, 0.8, then 1.2 deeper |

- **The two methods build opposite shapes.** EAGLE-3's global top-59 keeps only 3–6 first-layer
  nodes and widens with depth. TALON always keeps 10 (or 16) at depth 1 and narrows below.
- **TALON does reshape by workload, in the predicted direction:** narrow middle layers (5–6 nodes)
  and a deep tail on code; wide middle layers (10–12 nodes) and a short tail on QA and Sum. That
  reshaping does not translate into speed over a per-workload fixed tree.

## Earlier Path B findings (A100, H200, L40S)

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

### 9. TALON-S at every batch size, and the batch-size curve

`sglang_talon_bs1_32_h200/` repeats the batch-8 comparison at batch 1 and 32. TALON-S stays behind
EAGLE-3 at both: 2.51× vs 2.71× at batch 1 and 0.54× vs 0.56× at batch 32 (geometric means of the
six workloads), with the same 4–7% lower τ. The one exception is MT-Bench at batch 32 (+9%), where
τ is also lower, so it is likely scheduling noise rather than a real gain.

`sglang_bscurve_humaneval_alpaca_h200/` fills in batch 2, 4 and 16 for HumanEval and Alpaca. Speedup
over plain decoding, geometric mean of the two (batch 1, 8 and 32 from `sglang_h200/`):

| Shape | Batch 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| Chain, 3 deep | 2.50 | 2.47 | 2.44 | 2.36 | 2.22 | **1.99** |
| Chain, 7 deep | **3.01** | **3.01** | **2.92** | **2.80** | **2.50** | **1.99** |
| Tree 7 × 2 (budget 16) | 2.73 | 2.64 | 2.54 | 2.35 | 1.69 | 1.32 |
| Tree 6 × 3 (budget 32) | 2.71 | 2.68 | 2.48 | 2.05 | 1.35 | 0.91 |
| Tree 8 × 4 (budget 60) | 2.90 | 2.76 | 2.30 | 1.60 | 0.78 | 0.63 |
| Tree 9 × 10 (budget 60, EAGLE-3) | 2.95 | 2.84 | 2.34 | 1.64 | 1.00 | 0.63 |
| TALON-S (9 × 10) | – | 2.61 | 2.17 | – | 0.96 | – |
| SGLang adaptive | 2.74 | 2.66 | 2.66 | 2.37 | 2.20 | 1.62 |

Budget-60 trees break even at about batch 16. The 7-deep chain is best or tied at every batch size
on these two workloads, and TALON-S stays 5–8% behind EAGLE-3 at batch 2, 4 and 16 too.

### 10. How much of TALON fits SGLang's tree

`talon_shapes_*` reran TALON (N=60, mu=0.03, K=10) on all six workloads with every tree's shape
recorded: its depth, the most nodes expanded in one level, and the most children of one node.
SGLang's EAGLE-3 drafter (9 steps, top-k 10) keeps at most 10 nodes per level to expand and gives
each at most 10 children, up to 9 levels deep. `scripts/analyze_talon_shapes.py` counts the trees
that fit all three limits:

| Workload | Trees | Mean depth (max) | Depth ≤ 9 | ≤ 10 nodes expanded per level | ≤ 10 children per node | Fits SGLang |
| --- | --- | --- | --- | --- | --- | --- |
| HumanEval | 4,049 | 8.5 (21) | 63% | 69% | 76% | **26%** |
| GSM8K | 2,564 | 7.0 (18) | 85% | 59% | 73% | **32%** |
| MT-Bench | 10,274 | 6.9 (21) | 81% | 56% | 68% | **25%** |
| Alpaca | 2,852 | 6.4 (18) | 88% | 49% | 76% | **29%** |
| Sum | 4,216 | 6.0 (18) | 90% | 55% | 64% | **28%** |
| QA | 2,848 | 6.0 (19) | 89% | 57% | 62% | **26%** |

Only 25–32% of TALON's trees fit. Width is the main limit: about half the trees expand more than 10
nodes in some level (the widest level averages 18–23 nodes), and a quarter to a third give one
node more than 10 children, since TALON's gate keeps every child above the threshold with no
per-node limit. Depth matters mainly on HumanEval, where 37% of trees go past 9 levels. Fitting is
necessary for TALON-S to reproduce a tree but not sufficient, because SGLang also picks which nodes
to expand, so TALON-S matches TALON on at most a quarter to a third of steps. That is consistent
with TALON-S's 4–8% lower τ in sections 8 and 9.

The HumanEval, GSM8K and MT-Bench job ran on an A100-80GB rather than an H200. Tree shapes depend on
the drafter's probabilities, not the GPU (τ is within 0.3% of the H200 runs), so the table holds,
but that job's speeds are not comparable and are not used.

## Correctness

At T=0, speculative decoding should reproduce plain decoding exactly. Comparing token IDs on the A100
HumanEval and Alpaca runs, 89 to 92% of answers are identical under both EAGLE-3 and TALON. The
answers that differ mostly diverge hundreds of tokens in, and EAGLE-3 and TALON often diverge on the
same questions at the same token. That points to fp16 near-ties in the target model rather than a
bug in either tree builder.

`diagnose_mismatch_h200/` checks this on the H200 runs of all six workloads. For every answer that
differs from plain decoding, `scripts/diagnose_mismatch.py` finds the first differing token and
compares the target model's logits there. Every difference is an fp16 near-tie (the two tokens'
logits differ by at most 0.25, or one output stops a token earlier), or, on MT-Bench, a second turn
that inherits a difference from the first. None is suspicious:

| Workload | EAGLE-3 8 × 10 differing turns | TALON differing turns |
| --- | --- | --- |
| HumanEval | 4 (all near-ties) | 7 (all near-ties) |
| GSM8K | 9 (all near-ties) | 8 (all near-ties) |
| MT-Bench | 42 (24 near-ties, 2 stop-token near-ties, 16 inherited) | 45 (28 near-ties, 1 stop-token near-tie, 16 inherited) |
| Alpaca | 5 (all near-ties) | 5 (all near-ties) |
| Sum | 16 (all near-ties) | 20 (all near-ties) |
| QA | 10 (7 near-ties, 3 stop-token near-ties) | 8 (6 near-ties, 2 stop-token near-ties) |

## Caveats

- Path B is batch size 1 only. Path A covers batch sizes 1 to 32 with fixed shapes and SGLang's own drafter; TALON's rules there (TALON-S) work only within a fixed tree.
- The headline configurations (EAGLE-3 8 × 10, TALON, 6 × 10, 5 × 16) were run three times
  (`repeat2_*`, `repeat3_*`); the sweeps were run once. τ is identical across runs, and the same
  configuration's speed varies by 2–3%. One exception: in run 3 on HumanEval, GSM8K and MT-Bench,
  EAGLE-3 and plain decoding, which run first, were about 7% slower than in every other run, so
  that run overstates every other configuration relative to EAGLE-3 there. Across runs, TALON's
  and 6 × 10's gains on QA and Sum hold in all three (TALON +8% to +10% on QA, +3% to +6% on Sum;
  6 × 10 +4% to +6% on both). Gains on the other workloads are within run-to-run variance. The
  headline intervals resample questions, not runs.
- "Ours" was chosen on 17–29 calibration questions per workload. A larger or different calibration
  set might pick a different tree, and the Sum/QA gains are small enough to check on a repeat run.
- Path A ran SGLang with `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`, since its EAGLE-3 drafter declares a 2,048-token context. Whether any Sum prompt plus answer goes past 2,048 tokens has not been checked yet.
- 80 questions per benchmark. The intervals in each `summary.csv` compare a method with plain
  decoding; `summary_vs_eagle3.csv` has the paired intervals against EAGLE-3 (H200 only).
- Plain decoding in the EAGLE repo's code is slow (about 19 tok/s on the A100, 44 on the H200), so
  absolute speedups are higher than with an optimised serving engine. Comparisons between methods
  are unaffected.
- On the A100, part of the HumanEval run was split across two Colab sessions. The resume logic keeps
  a checkpoint only if the GPU and settings match.
