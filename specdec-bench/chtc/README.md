# CHTC TALON suite

Runs notebook cell 7's TALON ablations on HumanEval, Alpaca, and MT-Bench using
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

The model cache is downloaded into the job's scratch space. Completed and
partial results are synced to `/staging/$CHTC_USER/specdec-talon-three-bench-results`
every five minutes and restored when the job starts, so reruns continue from
the last completed question. Results use an isolated directory so they cannot
be mixed with outputs from the complete six-benchmark suite. Condor also
transfers `results/talon_three_bench` back to the submit directory when the job
exits.

## Time and resources

The suite runs 12 speculative configurations plus a plain autoregressive
baseline across three datasets (80 questions each; MT-Bench has two turns per
question), with up to 1,024 generated tokens per turn. That is 4,160
generation turns, or about 4.3 million output tokens at the maximum length.
Budget roughly 9–23 GPU-hours on an A100-class GPU, plus first-run package and
model downloads. Queue wait is separate and unpredictable. The estimate is
approximate because actual answer lengths and GPU throughput vary.

The submit file requests 8 CPUs, 48 GB RAM, 80 GB disk, and one GPU with at
least 40 GB VRAM. Results appear under `chtc/results/talon` on the CHTC login
side after Condor transfers them back. Job logs are in `chtc/logs/`.