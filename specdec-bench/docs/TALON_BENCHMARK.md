# TALON vs EAGLE-3 Benchmark Guide

This document explains how to run the TALON benchmark from the paper:
**"TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees"** (Liu et al., arXiv 2601.07353)

## Overview

TALON is a training-free, budget-driven adaptive tree expansion framework that:
- Uses **confidence-gated expansion** to build adaptive draft trees
- Creates **deep-and-narrow** trees for deterministic contexts
- Creates **shallow-and-wide** trees for uncertain contexts
- Achieves up to **5.16x speedup** over autoregressive decoding

## Quick Start (Batch Size 1)

```bash
cd specdec-bench
source .venv-eagle/bin/activate
python scripts/quickstart_talon.py
```

This runs a single prompt comparison:
- Plain AR decoding (baseline)
- Static EAGLE-3 (K=10, D=8, N=60)
- TALON (N=60, mu=0.03, K=10)

## Benchmark Phases

### Phase 0: Environment Check
```bash
nvidia-smi  # Need A100 (40GB) for batch sizes > 1
```

### Phase 1: Run Tests
```bash
source .venv-eagle/bin/activate
EAGLE_REPO=third_party/EAGLE python -m unittest discover -s tests
```

### Phase 2: TALON Tree Shape Analysis
Analyze how often TALON trees fit within SGLang's fixed envelope:
```bash
export EAGLE_DATA=$PWD/third_party/EAGLE/eagle/data
python scripts/run_talon_benchmark.py --data-dir $EAGLE_DATA --bench all --limit 10 \
    --talon --shape-analysis --out-dir results/talon_shape
```

### Phase 3: Smoke Test
Quick validation (10 questions, 2 benchmarks):
```bash
python scripts/run_talon_benchmark.py --data-dir $EAGLE_DATA --bench mt_bench,humaneval \
    --limit 10 --talon --static --with-ar --out-dir results/talon_smoke
```

### Phase 4: Full Benchmark
Complete benchmark suite (all 6 datasets):
```bash
# TALON with different configurations
for MU in 0.01 0.03 0.1; do
    python scripts/run_eagle.py --data-dir $EAGLE_DATA --bench all \
        --adaptive --budget 60 --mu $MU --root-k 10 \
        --out-dir results/talon/talon_n60_mu${MU}_k10 --skip-existing
done

# Static EAGLE-3 baseline
python scripts/run_eagle.py --data-dir $EAGLE_DATA --bench all --with-ar \
    --steps 9 --topk 10 --budget 60 \
    --out-dir results/talon/eagle3_baseline --skip-existing
```

### Phase 5: Generate Summary
```bash
python scripts/summarize.py results/talon --out results/talon/summary.csv
```

## Using the Suite Script

For convenience, use the all-in-one script:
```bash
bash scripts/run_talon_suite.sh results/talon 1  # Batch size 1
```

To run all phases including the full benchmark:
```bash
bash scripts/run_talon_suite.sh results/talon 1 --full
```

## Running on Google Colab

1. Upload `specdec-bench.zip` to `MyDrive/specdec/`
2. Add `HF_TOKEN` in Colab Secrets
3. Select A100 GPU runtime
4. Open `notebooks/colab_talon_benchmark.ipynb`
5. Run cells in order

## Configuration Parameters

### TALON (Algorithm 2)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `budget` (N) | 60 | Total verified nodes including root |
| `mu` | 0.03 | Confidence threshold for gating |
| `root_k` (K) | 10 | Width of forced top-K root layer |

### Static EAGLE-3 (Paper Baseline)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps` | 9 | Tree depth in layers (D+1) |
| `topk` (K) | 10 | Children per parent |
| `budget` (N) | 60 | Total draft tokens |

## Understanding Results

### Key Metrics
- **tau (τ)**: Mean accepted tokens per verify pass
- **delta (δ)**: Draft efficiency = draft_passes / verify_passes
- **speedup**: throughput / AR_throughput

### TALON vs EAGLE-3
From the paper:
- TALON achieves **lower delta** (fewer wasted draft passes)
- TALON adapts tree shape based on context difficulty
- Overall speedup is similar or better, with less compute waste

## Files Created

| File | Description |
|------|-------------|
| `scripts/run_talon_benchmark.py` | Main benchmark script |
| `scripts/quickstart_talon.py` | Quick single-prompt comparison |
| `scripts/run_talon_suite.sh` | Full benchmark suite |
| `configs/smoke_talon.yaml` | SGLang smoke test config |
| `configs/sweep_talon_sglang.yaml` | Full SGLang sweep config |
| `specdec/talon_sglang.py` | TALON for SGLang (Path A) |
| `specdec/sglang_talon_patch.py` | SGLang plugin |
| `notebooks/colab_talon_benchmark.ipynb` | Colab notebook |

## Known Limitations

1. **TALON-S in SGLang**: SGLang uses fixed tensor shapes, so TALON is capped at `topk` nodes per layer. When this cap binds, slots are filled by EAGLE-2's rule.

2. **Batch Size 1 Only for Full TALON**: The paper evaluates batch size 1. Path A (SGLang) gives TALON-S which has the drafting savings limitation.

3. **GPU Memory**: A100 (40GB) required for batch sizes > 1.

## Citation

```bibtex
@article{liu2026talon,
  title={TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees},
  author={Liu, Tianyu and Lv, Qitan and Shen, Yuhao and Sun, Xiao and Sun, Xiaoyan},
  journal={arXiv preprint arXiv:2601.07353},
  year={2026}
}
```
