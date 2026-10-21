"""Benchmark prompts and chat formatting.

Uses the six question files shipped in the EAGLE repo (eagle/data/<bench>/question.jsonl):
mt_bench (80, 2 turns each), humaneval, gsm8k, alpaca, sum (CNN/DM), qa (NQ): 80 each.
These are the same six benchmarks TALON and PRISM report on.

Prompt construction copies EAGLE's gen_ea_answer_llama3chat.py exactly:
  system message below + tokenizer.apply_chat_template(..., tokenize=False,
  add_generation_prompt=True), then tokenizer([prompt], add_special_tokens=False).
Both code paths (EAGLE repo and SGLang) use this function so the target sees
identical token ids.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Sequence, Tuple

BENCHES = ("mt_bench", "humaneval", "gsm8k", "alpaca", "sum", "qa")


def parse_benches(value: str) -> List[str]:
    """Expand 'all' or validate a comma-separated list of benchmark names."""
    benches = list(BENCHES) if value == "all" else [name.strip() for name in value.split(",")]
    if not benches or any(bench not in BENCHES for bench in benches):
        raise ValueError(f"benchmarks must be 'all' or names from {BENCHES}")
    if len(set(benches)) != len(benches):
        raise ValueError("benchmark list cannot contain duplicates")
    return benches

# Verbatim from SafeAILab/EAGLE eagle/evaluation/gen_ea_answer_llama3chat.py
# (note the two spaces after "safe.").
SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as "
    "possible, while being safe.  Your answers should not include any harmful, unethical, "
    "racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses "
    "are socially unbiased and positive in nature.\n\nIf a question does not make any sense, "
    "or is not factually coherent, explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information."
)


def load_questions(data_dir: str, bench: str) -> List[Dict]:
    """Load data_dir/<bench>/question.jsonl. data_dir is EAGLE/eagle/data."""
    path = os.path.join(data_dir, bench, "question.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; point --data-dir at EAGLE/eagle/data")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if "turns" not in q or not q["turns"]:
                raise ValueError(f"{path}: question {q.get('question_id')} has no turns")
            out.append({
                "question_id": q["question_id"],
                "category": q.get("category", bench),
                "turns": list(q["turns"]),
                "bench": bench,
            })
    return out


def _bucket(bench: str, qid) -> float:
    h = hashlib.sha256(f"{bench}:{qid}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split(questions: Sequence[Dict], calib_frac: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
    """Deterministic calibration/test split by hashing (bench, question_id).

    Stable across machines and independent of file order, so the split can be
    frozen before any sweep.
    """
    calib, test = [], []
    for q in questions:
        (calib if _bucket(q["bench"], q["question_id"]) < calib_frac else test).append(q)
    return calib, test


def select(questions: List[Dict], subset: str, limit: int = 0) -> List[Dict]:
    if subset == "all":
        qs = list(questions)
    elif subset in ("calib", "test"):
        calib, test = split(questions)
        qs = calib if subset == "calib" else test
    else:
        raise ValueError("subset must be all|calib|test")
    return qs[:limit] if limit and limit > 0 else qs


def initial_messages() -> List[Dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def encode_messages(tokenizer, messages: List[Dict[str, str]]) -> List[int]:
    """Exactly the EAGLE eval-script encoding."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return list(tokenizer([prompt], add_special_tokens=False).input_ids[0])


def stop_token_ids(tokenizer) -> List[int]:
    """Same stop set the EAGLE eval script truncates at."""
    ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    return [i for i in dict.fromkeys(ids) if i is not None]


def truncate_at_stop(output_ids: Sequence[int], stops: Sequence[int]) -> List[int]:
    out = list(output_ids)
    for i, t in enumerate(out):
        if t in stops:
            return out[:i]
    return out
