# Adaptive tree (TALON) vs EAGLE-3: results so far

All runs use Llama-3.1-8B-Instruct as the target and the official EAGLE-3 drafter
(`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`), batch size 1, greedy decoding (T=0), up to 1,024 new
tokens per turn and 80 questions per benchmark (MT-Bench has two turns per question). Each folder
below is one run of `scripts/run_adaptive_suite.sh` (cell 7 of the Colab notebook, or the CHTC job):

- **EAGLE-3:** fixed tree, K=10 children per node, depth 8 (`steps=9`), at budgets N = 32, 48, 60, 96.
- **TALON (adaptive tree):** the same budgets with mu=0.03, plus mu = 0.01, 0.02, 0.04, 0.05 at N=60.
- **Plain decoding:** the speedup baseline, on the same model load.

| Folder | GPU | Benchmarks |
| --- | --- | --- |
| `adaptive_three_bench/` | Colab A100 | Full suite: GSM8K, Sum, QA. Main comparison only (N=60): HumanEval, Alpaca |
| `chtc_h200/adaptive_three_bench/` | CHTC H200 | Full suite: HumanEval, Alpaca, MT-Bench |
| `chtc_l40s/adaptive_gsm8k_sum_qa/` | CHTC L40S | Full suite: GSM8K, Sum, QA |

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

## Correctness

At T=0, speculative decoding should reproduce plain decoding exactly. Comparing token IDs on the A100
HumanEval and Alpaca runs, 89 to 92% of answers are identical under both EAGLE-3 and TALON. The
answers that differ mostly diverge hundreds of tokens in, and EAGLE-3 and TALON often diverge on the
same questions at the same token. That points to fp16 near-ties in the target model rather than a
bug in either tree builder. `scripts/diagnose_mismatch.py` classifies each mismatch and has not yet
been run on these files.

## Caveats

- Batch size 1 only. The SGLang sweep at batch sizes 8 and 32 has not been run yet.
- 80 questions per benchmark. Speedup confidence intervals in `summary.csv` compare each method with
  plain decoding, not TALON with EAGLE-3 directly. A paired bootstrap would be needed to call the
  A100 QA/Sum gains significant.
- Plain decoding in the EAGLE repo's code is slow (about 19 tok/s on the A100, 44 on the H200), so
  absolute speedups are higher than with an optimised serving engine. Comparisons between methods
  are unaffected.
- On the A100, part of the HumanEval run was split across two Colab sessions. The resume logic keeps
  a checkpoint only if the GPU and settings match.
