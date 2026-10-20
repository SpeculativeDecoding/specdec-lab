#!/usr/bin/env bash
# Path A environment: SGLang from a pinned release tag + this harness's client deps.
#   SGLANG_TAG=v0.5.22  needs a CUDA 13 driver (torch 2.14.1, flashinfer cu13)
#   SGLANG_TAG=v0.5.19  last release for CUDA 12 drivers (torch 2.13.0)
# The harness was checked against both tags (same flags, tree semantics, meta_info).
# Check your driver first:  nvidia-smi  (top-right "CUDA Version").
set -euo pipefail
SGLANG_TAG=${SGLANG_TAG:-v0.5.22}
ENV_DIR=${ENV_DIR:-$PWD/.venv-sglang}
WORK=${WORK:-$PWD/third_party}

# needs Python >= 3.11
# Colab and some Debian/Ubuntu images ship Python without ensurepip; fall back to virtualenv.
python3 -m venv "$ENV_DIR" 2>/dev/null || { python3 -m pip install -q virtualenv && python3 -m virtualenv --clear "$ENV_DIR"; }
source "$ENV_DIR/bin/activate"
pip install --upgrade pip uv
mkdir -p "$WORK"
if [ ! -d "$WORK/sglang" ]; then
  git clone --branch "$SGLANG_TAG" --depth 1 https://github.com/sgl-project/sglang.git "$WORK/sglang"
fi
cd "$WORK/sglang"
uv pip install -e "python" --prerelease=allow
cd - >/dev/null
uv pip install pyyaml requests numpy
python -c "import sglang, transformers; print('sglang', sglang.__version__, '| transformers', transformers.__version__)"

# Alternative without building: the official image for the same tag, e.g.
#   docker run --gpus all --shm-size 32g --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v $PWD:/work -it lmsysorg/sglang:${SGLANG_TAG} bash      (use :v0.5.19-cu129 on CUDA 12)
# then: pip install pyyaml  and run the scripts from /work.
