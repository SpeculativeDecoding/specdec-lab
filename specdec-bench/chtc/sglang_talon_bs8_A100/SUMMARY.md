# EAGLE-3 vs TALON-S at batch size 8 on A100 (SGLang v0.5.22)

Run of `configs/talon_vs_eagle3_bs8.yaml` (commit `537b441`, branch `feat/talon-sglang-bs8`) on a local
2x A100-SXM4-40GB machine, 2026-10-24. Llama-3.1-8B-Instruct + `lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B`,
fp16, T=0, batch 8, all questions of the six EAGLE benchmarks, up to 1024 new tokens per turn.

Both speculative servers use the same 9-step x 10-child tree with a budget of 60 verified tokens, so drafting
cost is identical and only the node-selection rule differs:

- **EAGLE-3** (`eagle3_s9_k10_b60_eagle3_ref`): SGLang's EAGLE-2 rule, global top-59 by cumulative probability.
- **TALON-S** (`talon_s9_k10_b60_mu0.03`): TALON's rules via `specdec/sglang_talon_plugin.py`, mu = 0.03.

## Headline

**EAGLE-3 is faster than TALON-S on all six benchmarks.** Geometric-mean speedup over plain decoding is
**1.36x for EAGLE-3 vs 1.26x for TALON-S**; TALON-S delivers 91-94% of EAGLE-3's throughput. All 18 runs
finished with 0 failed requests, and the TALON-S hook was verified to run (`talon_marker.jsonl`).

| Benchmark | Plain (tok/s) | EAGLE-3 (tok/s) | TALON-S (tok/s) | EAGLE-3 speedup | TALON-S speedup | EAGLE-3 tau | TALON-S tau |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MT-Bench  | 561 | **779** | 714 | **1.39x** | 1.27x | **6.47** | 6.09 |
| HumanEval | 573 | **854** | 799 | **1.49x** | 1.39x | **7.04** | 6.75 |
| GSM8K     | 568 | **765** | 710 | **1.35x** | 1.25x | **6.43** | 6.10 |
| Alpaca    | 572 | **833** | 755 | **1.46x** | 1.32x | **6.89** | 6.36 |
| Sum       | 428 | **569** | 531 | **1.33x** | 1.24x | **5.48** | 5.18 |
| QA        | 565 | **663** | 617 | **1.17x** | 1.09x | **5.56** | 5.25 |

Throughput is `throughput_full_occupancy` (output tokens / seconds while all 8 slots were busy), the value
`scripts/summarize.py` uses; speedup is against plain decoding on the same benchmark. `tau` = generated tokens
per verify pass, bonus token included. Full numbers: `summary.csv`.

## Where the gap comes from

Throughput per request is `tau / step time`, so the TALON-S / EAGLE-3 throughput ratio splits into an
acceptance term and a per-step cost term (step time = SGLang's median decode step at batch 8):

| Benchmark | Throughput ratio | tau ratio | Step time EAGLE-3 / TALON-S (ms) | Step-time overhead |
| --- | ---: | ---: | ---: | ---: |
| MT-Bench  | 0.918 | 0.942 | 65.2 / 66.1 | +1.5% |
| HumanEval | 0.936 | 0.959 | 65.1 / 65.9 | +1.2% |
| GSM8K     | 0.928 | 0.950 | 64.9 / 65.9 | +1.5% |
| Alpaca    | 0.906 | 0.923 | 64.6 / 65.8 | +1.7% |
| Sum       | 0.934 | 0.946 | 65.1 / 66.6 | +2.2% |
| QA        | 0.930 | 0.945 | 64.9 / 66.2 | +1.9% |

- **Most of the loss is acceptance, not overhead.** TALON-S accepts 4-8% fewer tokens per verify pass; the
  plugin's extra scoring work costs only 1-2% per step (about 1 ms on a 65 ms step). The two terms multiply
  to within about 1 point of the measured throughput ratio.
- **The tau loss is consistent per request, not driven by a few outliers.** TALON-S has a lower per-turn tau
  than EAGLE-3 on 136/160 MT-Bench turns, 73/80 HumanEval, 58/80 GSM8K, 62/80 Alpaca, 70/80 Sum, and 59/80 QA.
- **Why acceptance drops (hypothesis, not verified):** with a fixed 60-token budget and fixed tree shape,
  EAGLE-2's global top-59 by cumulative draft probability already picks the nodes the drafter thinks are most
  likely to be accepted. TALON-S's forced top-10 root, confidence gate, and layer-first budget spend part of
  that same budget on shallow, lower-probability nodes, which shortens the deepest accepted paths. TALON's
  reported gains come partly from adapting tree shape and size per step; this SGLang port keeps shapes fixed so
  CUDA graphs still apply, which may remove the part of TALON that helps. Comparing per-step accepted depth
  (not recorded here; `mean_tree_depth` is empty for Path A) or sweeping `mu` would test this.

## Other observations

- **Speculation pays off on every benchmark, but QA least.** A batch-8 verify step costs about 4.7x a plain
  decode step (65 ms vs 13.9 ms), so a method needs tau above ~4.7 to win; QA and Sum have the lowest tau
  (5.2-5.6) and the smallest speedups.
- **Benchmark ranking is the same for both methods** (HumanEval > Alpaca > MT-Bench > GSM8K > Sum > QA by
  speedup), matching the earlier finding that the workload effect within a budget is real but small.
- **Outputs are not bit-identical to plain decoding.** The share of turns whose text exactly matches plain
  decoding is 72-95% for EAGLE-3 and 72-93% for TALON-S, with similar total output lengths (within 1%).
  Since both methods diverge from plain decoding by similar amounts, this looks like fp16 near-tie flips and
  batch-dependent kernels (the README notes this at T=0) rather than a TALON-S correctness bug; MT-Bench is
  lowest because a flip in turn 1 changes the turn 2 prompt. It has not been checked token by token.

## How this run differs from the config (read before comparing)

- **GPU:** A100-40GB, not the H200 used for `results/sglang_h200`. Compare these rows with each other only.
- **`mem_fraction_static`:** 0.70 for both speculative servers instead of 0.80. At 0.80 the 9 x 10 tree's
  CUDA graphs ran out of memory on 40 GB during server start. This only shrinks the KV cache, which batch 8
  does not come close to filling. Plain decoding ran at 0.80. The exact configs used are in `configs/`.
- **Two GPUs:** plain decoding and EAGLE-3 ran on GPU 0, TALON-S on GPU 1 (same model, both idle otherwise),
  each server on its own port.
- **Plugin registration:** without pip, as in `chtc/job_sglang.sh` (a dist-info with the `sglang.srt.plugins`
  entry point on `PYTHONPATH`). Environment: conda `sglang`, SGLang 0.5.22.

## Files

- `summary.csv`: all 18 runs (from `scripts/summarize.py`).
- `ar/`, `eagle3_s9_k10_b60_eagle3_ref/`, `talon_s9_k10_b60_mu0.03/`: per-benchmark `.jsonl` (a `_meta` /
  `_summary` header line, then one line per turn) and the server log. `talon_marker.jsonl` shows the hook
  registered in each server process and was called.
- `sweep.log` (plain decoding), `sweep_gpu0_eagle3.log`, `sweep_gpu1_talon.log`: sweep driver logs.
- `configs/`: the A100 configs actually used for the EAGLE-3 and TALON-S servers.
