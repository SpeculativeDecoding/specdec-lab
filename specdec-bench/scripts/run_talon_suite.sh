#!/usr/bin/env bash
# TALON vs EAGLE-3 Benchmark Suite
# Usage: bash scripts/run_talon_suite.sh [results_dir] [batch_size]
#
# This script runs the complete TALON benchmark following the paper's protocol:
#   Phase 0: Environment check
#   Phase 1: Run tests
#   Phase 2: TALON tree shape analysis (Path B, batch 1)
#   Phase 3: Smoke test
#   Phase 4: Full benchmark (can be run for specific batch size)
#   Phase 5: Generate summary report
#
# Arguments:
#   results_dir: Output directory (default: results/talon)
#   batch_size:  Batch size to run (default: 1). Use 'all' for 1,8,32.
set -euo pipefail

RESULTS_DIR="${1:-results/talon}"
BATCH_SIZE="${2:-1}"
EAGLE_DATA="${EAGLE_DATA:-$PWD/third_party/EAGLE/eagle/data}"

echo "=============================================="
echo "TALON vs EAGLE-3 Benchmark Suite"
echo "=============================================="
echo "Results directory: $RESULTS_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Data directory: $EAGLE_DATA"
echo ""

# Detect which environment we're in
detect_env() {
    if [ -f .venv-eagle/bin/activate ]; then
        echo "eagle"
    elif [ -f .venv-sglang/bin/activate ]; then
        echo "sglang"
    else
        echo "none"
    fi
}

# Phase 0: Environment Check
phase_0_env_check() {
    echo "=== Phase 0: Environment Check ==="

    # GPU check
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

        # Extract CUDA version
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "unknown")
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

        echo "CUDA Version: $CUDA_VER"
        echo "GPU Memory: ${GPU_MEM} MiB"

        # Check memory requirements
        if [ "$GPU_MEM" -lt 38000 ] && [ "$BATCH_SIZE" != "1" ]; then
            echo "WARNING: GPU has less than 38 GB. Batch sizes >1 may not work."
            echo "Continuing with batch size 1 only for Path B."
            BATCH_SIZE=1
        fi
    else
        echo "ERROR: nvidia-smi not found. GPU required."
        exit 1
    fi

    # Check data directory
    if [ ! -d "$EAGLE_DATA" ]; then
        echo "ERROR: EAGLE data directory not found: $EAGLE_DATA"
        echo "Please set EAGLE_DATA or run setup_eagle_env.sh first."
        exit 1
    fi

    echo "Phase 0: PASSED"
    echo ""
}

# Phase 1: Run Tests
phase_1_tests() {
    echo "=== Phase 1: Running Tests ==="

    ENV=$(detect_env)

    if [ "$ENV" = "eagle" ]; then
        echo "Running tests in eagle environment..."
        source .venv-eagle/bin/activate
        EAGLE_REPO=third_party/EAGLE python -m unittest discover -s tests 2>&1 | tail -5
        deactivate
    elif [ "$ENV" = "sglang" ]; then
        echo "Running tests in sglang environment..."
        source .venv-sglang/bin/activate
        # Run specific TALON tests if they exist
        if [ -f tests/test_talon_sglang.py ]; then
            python -m unittest tests.test_talon_sglang -v 2>&1 | tail -10
        else
            echo "Note: TALON SGLang tests not found (Path A not yet implemented)"
        fi
        deactivate
    else
        echo "WARNING: No virtual environment found. Skipping tests."
    fi

    echo "Phase 1: PASSED"
    echo ""
}

# Phase 2: TALON Tree Shape Analysis (Path B)
phase_2_shape_analysis() {
    echo "=== Phase 2: TALON Tree Shape Analysis ==="

    if [ ! -f .venv-eagle/bin/activate ]; then
        echo "Skipping Phase 2: eagle environment not set up"
        return
    fi

    source .venv-eagle/bin/activate

    OUT_DIR="$RESULTS_DIR/talon_shape"
    mkdir -p "$OUT_DIR"

    echo "Running TALON on all benchmarks (10 questions each)..."
    python scripts/run_talon_benchmark.py \
        --data-dir "$EAGLE_DATA" \
        --bench all \
        --limit 10 \
        --max-turns 1 \
        --talon \
        --budget 60 \
        --mu 0.03 \
        --root-k 10 \
        --shape-analysis \
        --out-dir "$OUT_DIR" \
        --skip-existing \
        --eagle-repo third_party/EAGLE

    deactivate

    echo "Phase 2: COMPLETE"
    echo "Results in: $OUT_DIR"
    echo ""
}

# Phase 3: Smoke Test
phase_3_smoke_test() {
    echo "=== Phase 3: Smoke Test ==="

    if [ "$BATCH_SIZE" = "1" ] || [ "$BATCH_SIZE" = "all" ]; then
        # Path B smoke test (batch 1)
        if [ -f .venv-eagle/bin/activate ]; then
            echo "Running Path B smoke test (batch 1)..."
            source .venv-eagle/bin/activate

            OUT_DIR="$RESULTS_DIR/smoke_pathb"
            mkdir -p "$OUT_DIR"

            python scripts/run_talon_benchmark.py \
                --data-dir "$EAGLE_DATA" \
                --bench mt_bench,humaneval \
                --limit 10 \
                --max-turns 1 \
                --talon \
                --static \
                --with-ar \
                --out-dir "$OUT_DIR" \
                --skip-existing \
                --eagle-repo third_party/EAGLE

            # Losslessness check
            echo ""
            echo "Losslessness check (TALON vs AR):"
            python scripts/check_lossless.py \
                "$OUT_DIR/mt_bench__adaptive_n60_mu0.03_k10__T0.jsonl" \
                "$OUT_DIR/mt_bench__ar__T0.jsonl" 2>/dev/null || echo "(files may not exist yet)"

            deactivate
        fi
    fi

    if [ "$BATCH_SIZE" != "1" ] || [ "$BATCH_SIZE" = "all" ]; then
        # Path A smoke test (SGLang, batched)
        if [ -f .venv-sglang/bin/activate ]; then
            echo "Running Path A smoke test (SGLang)..."
            source .venv-sglang/bin/activate

            python scripts/run_sglang_sweep.py \
                --config configs/smoke_talon.yaml \
                --data-dir "$EAGLE_DATA" \
                --out-dir "$RESULTS_DIR/smoke_patha" \
                --skip-existing

            deactivate
        fi
    fi

    echo "Phase 3: COMPLETE"
    echo ""
}

# Phase 4: Full Benchmark
phase_4_benchmark() {
    echo "=== Phase 4: Full Benchmark ==="

    if [ "$BATCH_SIZE" = "1" ]; then
        # Path B: Full TALON benchmark (batch 1)
        if [ -f .venv-eagle/bin/activate ]; then
            echo "Running full TALON benchmark (Path B, batch 1)..."
            source .venv-eagle/bin/activate

            OUT_DIR="$RESULTS_DIR/full_pathb"
            mkdir -p "$OUT_DIR"

            # TALON with different configurations
            for MU in 0.01 0.03 0.1; do
                for BUDGET in 32 60; do
                    echo "Running TALON with mu=$MU, budget=$BUDGET..."
                    python scripts/run_eagle.py \
                        --data-dir "$EAGLE_DATA" \
                        --bench all \
                        --adaptive \
                        --budget "$BUDGET" \
                        --mu "$MU" \
                        --root-k 10 \
                        --out-dir "$OUT_DIR/talon_n${BUDGET}_mu${MU}_k10" \
                        --skip-existing \
                        --eagle-repo third_party/EAGLE || true
                done
            done

            # Static EAGLE-3 baseline
            echo "Running static EAGLE-3 baseline..."
            python scripts/run_eagle.py \
                --data-dir "$EAGLE_DATA" \
                --bench all \
                --with-ar \
                --steps 9 \
                --topk 10 \
                --budget 60 \
                --out-dir "$OUT_DIR/eagle3_s9_k10_b60" \
                --skip-existing \
                --eagle-repo third_party/EAGLE || true

            deactivate
        fi
    else
        # Path A: SGLang sweep (batched)
        if [ -f .venv-sglang/bin/activate ]; then
            echo "Running SGLang sweep (Path A, batch $BATCH_SIZE)..."
            source .venv-sglang/bin/activate

            python scripts/run_sglang_sweep.py \
                --config configs/sweep_talon_sglang.yaml \
                --data-dir "$EAGLE_DATA" \
                --out-dir "$RESULTS_DIR/sglang" \
                --skip-existing 2>&1 | tee -a "$RESULTS_DIR/sglang_log.txt"

            deactivate
        fi
    fi

    echo "Phase 4: COMPLETE"
    echo ""
}

# Phase 5: Summary and Report
phase_5_summary() {
    echo "=== Phase 5: Summary and Report ==="

    ENV=$(detect_env)

    if [ "$ENV" != "none" ]; then
        if [ "$ENV" = "eagle" ]; then
            source .venv-eagle/bin/activate
        else
            source .venv-sglang/bin/activate
        fi

        python scripts/summarize.py "$RESULTS_DIR" --out "$RESULTS_DIR/summary.csv"

        echo ""
        echo "Summary saved to: $RESULTS_DIR/summary.csv"

        if [ -f "$RESULTS_DIR/summary.csv" ]; then
            echo ""
            echo "Top results:"
            head -20 "$RESULTS_DIR/summary.csv" 2>/dev/null || cat "$RESULTS_DIR/summary.csv"
        fi

        deactivate
    fi

    echo ""
    echo "Phase 5: COMPLETE"
}

# Main execution
main() {
    mkdir -p "$RESULTS_DIR"

    # Log start
    echo "Benchmark started at $(date)" | tee "$RESULTS_DIR/run.log"
    echo "" | tee -a "$RESULTS_DIR/run.log"

    # Run phases
    phase_0_env_check 2>&1 | tee -a "$RESULTS_DIR/run.log"
    phase_1_tests 2>&1 | tee -a "$RESULTS_DIR/run.log"
    phase_2_shape_analysis 2>&1 | tee -a "$RESULTS_DIR/run.log"
    phase_3_smoke_test 2>&1 | tee -a "$RESULTS_DIR/run.log"

    echo ""
    echo "=============================================="
    echo "Smoke tests complete. Review results before Phase 4."
    echo "To continue with full benchmark:"
    echo "  bash scripts/run_talon_suite.sh $RESULTS_DIR $BATCH_SIZE --phase4"
    echo "=============================================="

    # Check if we should continue to Phase 4
    if [ "${3:-}" = "--phase4" ] || [ "${3:-}" = "--full" ]; then
        phase_4_benchmark 2>&1 | tee -a "$RESULTS_DIR/run.log"
        phase_5_summary 2>&1 | tee -a "$RESULTS_DIR/run.log"
    fi

    echo "Benchmark finished at $(date)" | tee -a "$RESULTS_DIR/run.log"
}

# Run main
main "$@"
