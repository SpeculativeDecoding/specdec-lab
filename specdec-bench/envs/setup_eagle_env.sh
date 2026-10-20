#!/usr/bin/env bash
# Path B environment: official EAGLE repo + this harness. Run from the specdec-bench root.
set -euo pipefail
EAGLE_COMMIT=cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b   # SafeAILab/EAGLE main, 19 Feb 2026 (verified)
ENV_DIR=${ENV_DIR:-$PWD/.venv-eagle}
WORK=${WORK:-$PWD/third_party}

# Colab and some Debian/Ubuntu images ship Python without ensurepip; fall back to virtualenv.
python3 -m venv "$ENV_DIR" 2>/dev/null || { python3 -m pip install -q virtualenv && python3 -m virtualenv --clear "$ENV_DIR"; }
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r envs/requirements-eagle.txt

mkdir -p "$WORK"
if [ ! -d "$WORK/EAGLE" ]; then
  git clone https://github.com/SafeAILab/EAGLE.git "$WORK/EAGLE"
fi
git -C "$WORK/EAGLE" checkout "$EAGLE_COMMIT"
# NOTE: do NOT `pip install -e` EAGLE normally: its setup.py pins torch==2.0.1 and
# transformers==4.46.2 and would downgrade both. Install it without dependencies.
pip install --no-deps -e "$WORK/EAGLE"

python - <<'PY'
import torch, transformers
from eagle.model.ea_model import EaModel  # fails on transformers>=4.54 (LossKwargs)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| transformers", transformers.__version__, "| EaModel import OK")
PY
echo "export EAGLE_REPO=$WORK/EAGLE   # data dir: \$EAGLE_REPO/eagle/data"
