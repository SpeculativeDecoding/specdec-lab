#!/usr/bin/env bash
# Smallest SGLang check: serve Llama-3.1-8B-Instruct with EAGLE-3 and send one request.
# Usage: bash scripts/quickstart_sglang.sh   (inside the SGLang environment)
set -euo pipefail
PORT=${PORT:-30000}
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
  --speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 60 \
  --dtype float16 --max-running-requests 8 --cuda-graph-max-bs-decode 8 \
  --mem-fraction-static 0.8 --port "$PORT" > sglang_quickstart.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
echo "waiting for server (log: sglang_quickstart.log)..."
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died; see sglang_quickstart.log"; exit 1; }
  sleep 5
done
python - "$PORT" <<'PY'
import sys, requests
from transformers import AutoTokenizer
port = sys.argv[1]
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
msgs = [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome."}]
ids = tok([tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)],
          add_special_tokens=False).input_ids[0]
r = requests.post(f"http://127.0.0.1:{port}/generate",
                  json={"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 256}}).json()
mi = r["meta_info"]
print(r["text"])
print(f"\ncompletion_tokens={mi['completion_tokens']}  verify passes={mi.get('spec_verify_ct')}  "
      f"tau (spec_accept_length)={mi.get('spec_accept_length')}")
PY
