# CHTC adaptive tree suite

Runs notebook cell 7's adaptive tree ablations on HumanEval, Alpaca, and MT-Bench using
one GPU with at least 40 GB of VRAM. These represent focused code generation,
general single-turn instructions, and broad multi-turn dialogue. The job uses
a CUDA 12.4 PyTorch container and requires access to the gated
`meta-llama/Llama-3.1-8B-Instruct` Hugging Face model.

## Configure and submit

Create `chtc/.env` locally (it is ignored by Git):

```sh
CHTC_USER=your_chtc_username
HF_TOKEN=your_huggingface_read_token
```

Accept the Llama model license on Hugging Face for the account associated with
the token, then run from this folder or any local directory:

```sh
bash chtc/upload_and_submit.sh
```

The helper packages the project, copies it to CHTC staging, and submits the
job from the CHTC login node. Set `CHTC_HOST` in `chtc/.env` only if a different
login host is needed.

The project archive and credentials are spooled to the Condor scheduler at
submission; the model cache and partial results stay in the job's scratch space
while it runs. Results use an isolated directory so they cannot be mixed with
outputs from the complete six-benchmark suite. Since spooled jobs keep their
outputs at the scheduler, retrieve them from the CHTC access point after the
job finishes:

```sh
condor_transfer_data CLUSTER_ID
```

If the job is evicted, its scratch checkpoints are not persistent and the
interrupted configuration must be rerun.

## Time and resources

The full suite remains available as `bash scripts/run_adaptive_suite.sh`, which
runs all six benchmarks by default. For a custom selection, set `BENCHES` to a
comma-separated list. The CHTC job uses
`bash scripts/run_adaptive_three_bench_suite.sh` to run only the three benchmarks
listed above.

The suite runs 12 speculative configurations plus a plain autoregressive
baseline across three datasets (80 questions each; MT-Bench has two turns per
question), with up to 1,024 generated tokens per turn. That is 4,160
generation turns, or about 4.3 million output tokens at the maximum length.
Budget roughly 9–23 GPU-hours on an A100-class GPU, plus first-run package and
model downloads. Queue wait is separate and unpredictable. The estimate is
approximate because actual answer lengths and GPU throughput vary.

The submit file requests 8 CPUs, 48 GB RAM, 80 GB disk, and one GPU with at
least 40 GB VRAM. Results appear under `chtc/results/adaptive` on the CHTC login
side after Condor transfers them back. Job logs are in `chtc/logs/`.