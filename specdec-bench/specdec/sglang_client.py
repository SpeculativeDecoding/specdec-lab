"""Closed-loop load generator for SGLang's native /generate endpoint.

"Batch size b" = at most b conversations in flight (one worker thread each).
Each worker takes the next question, runs all its turns in order (feeding the
answer back, like the EAGLE script), then takes another question.

Fields read from the response (sglang v0.5.22, managers/tokenizer_manager.py):
  meta_info.completion_tokens, meta_info.prompt_tokens, meta_info.finish_reason,
  and for speculative runs meta_info.spec_verify_ct and meta_info.spec_accept_length
  (= completion_tokens / spec_verify_ct, bonus token included).
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Sequence

from .data import encode_messages, initial_messages
from .metrics import full_occupancy_window, overlap_throughput


def _post(session, url: str, payload: Dict, timeout: float) -> Dict:
    r = session.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def run_load(base_url: str, tokenizer, questions: Sequence[Dict], concurrency: int, *,
             max_new_tokens: int = 1024, temperature: float = 0.0, max_turns: int = 0,
             timeout: float = 1800.0, session_factory=None) -> Dict:
    """Send every question (all turns) with at most `concurrency` in flight.

    Returns {"rows": per-turn records, "spans": per-conversation spans, "window": ...}.
    """
    import requests

    session_factory = session_factory or requests.Session
    work: "queue.Queue[Dict]" = queue.Queue()
    for q in questions:
        work.put(dict(q, turns=q["turns"][:max_turns]) if max_turns else q)

    rows: List[Dict] = []
    spans: List[Dict] = []
    errors: List[str] = []
    lock = threading.Lock()
    url = base_url.rstrip("/") + "/generate"

    def worker():
        sess = session_factory()
        while True:
            try:
                q = work.get_nowait()
            except queue.Empty:
                return
            messages = initial_messages()
            conv_start = time.perf_counter()
            conv_tokens = 0
            for j, turn in enumerate(q["turns"]):
                messages.append({"role": "user", "content": turn})
                ids = encode_messages(tokenizer, messages)
                payload = {"input_ids": ids,
                           "sampling_params": {"temperature": float(temperature),
                                               "max_new_tokens": int(max_new_tokens)}}
                t0 = time.perf_counter()
                try:
                    out = _post(sess, url, payload, timeout)
                except Exception as e:  # keep going; report at the end
                    with lock:
                        errors.append(f'{q["bench"]}:{q["question_id"]}:{j}: {e!r}')
                    break
                t1 = time.perf_counter()
                mi = out.get("meta_info", {})
                text = out.get("text", "")
                messages.append({"role": "assistant", "content": text.strip()})
                n_out = int(mi.get("completion_tokens", 0))
                conv_tokens += n_out
                rec = {
                    "bench": q["bench"], "question_id": q["question_id"], "turn": j,
                    "qid_turn": f'{q["bench"]}:{q["question_id"]}:{j}',
                    "text": text, "output_tokens": n_out,
                    "prompt_tokens": int(mi.get("prompt_tokens", len(ids))),
                    "finish_reason": mi.get("finish_reason"),
                    "wall_time": t1 - t0, "start": t0, "end": t1,
                }
                if "spec_verify_ct" in mi:
                    rec["verify_steps"] = int(mi["spec_verify_ct"])
                    rec["new_tokens"] = n_out
                    rec["tau"] = float(mi.get("spec_accept_length",
                                              n_out / max(1, rec["verify_steps"])))
                with lock:
                    rows.append(rec)
            with lock:
                spans.append({"start": conv_start, "end": time.perf_counter(),
                              "output_tokens": conv_tokens})

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_end = time.perf_counter()

    total_tokens = sum(r["output_tokens"] for r in rows)
    window = full_occupancy_window(spans, concurrency)
    summary = {
        "concurrency": concurrency,
        "n_turns": len(rows),
        "n_errors": len(errors),
        "errors": errors[:20],
        "total_output_tokens": total_tokens,
        "elapsed_s": t_end - t_start,
        "throughput_all": total_tokens / (t_end - t_start) if t_end > t_start else float("nan"),
        "throughput_full_occupancy": (overlap_throughput(spans, *window) if window else None),
        "full_occupancy_window_s": (window[1] - window[0]) if window else None,
    }
    spec_rows = [r for r in rows if "verify_steps" in r]
    if spec_rows:
        summary["tau"] = (sum(r["new_tokens"] for r in spec_rows)
                          / max(1, sum(r["verify_steps"] for r in spec_rows)))
    return {"rows": rows, "spans": spans, "summary": summary}


def server_step_times(base_url: str, concurrency: int) -> Optional[Dict]:
    """Per-step decode time recorded by the server for this running-batch size.

    Requires SGLANG_RECORD_STEP_TIME=1 (set by Server). Returns 20th percentile and
    median, the same statistic SGLang's scripts/playground/bench_speculative.py uses.
    """
    import numpy as np
    import requests
    try:
        info = requests.get(base_url.rstrip("/") + "/server_info", timeout=30).json()
        d = info["internal_states"][0].get("step_time_dict", {})
        xs = d.get(str(concurrency)) or d.get(concurrency)
        if not xs:
            return None
        return {"p20": float(np.percentile(xs, 20)), "median": float(np.median(xs)), "n": len(xs)}
    except Exception:
        return None


def flush_cache(base_url: str) -> None:
    import requests
    try:
        requests.post(base_url.rstrip("/") + "/flush_cache", timeout=60)
    except Exception:
        pass


def write_run(out_path: str, meta: Dict, result: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp_path = out_path + ".partial"     # rename at the end: never a half-written result
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta, "_summary": result["summary"]}) + "\n")
        for r in result["rows"]:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp_path, out_path)
