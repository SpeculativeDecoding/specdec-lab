"""Metrics shared by both code paths. Pure Python + numpy (no torch).

Definitions (stated once, used everywhere):
  tau           = generated tokens / verify passes. Includes the bonus token, so
                  tau >= 1. Matches EAGLE (new_token / (idx+1)) and SGLang's
                  meta_info["spec_accept_length"] = completion_tokens / spec_verify_ct.
  throughput    = output tokens (truncated at the first stop token) / wall seconds.
  speedup       = throughput(method) / throughput(plain decoding) on the same prompts.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np


def tau(new_tokens: int, verify_steps: int) -> float:
    if verify_steps <= 0:
        raise ValueError("verify_steps must be positive")
    return new_tokens / verify_steps


def pooled_tau(records: Iterable[Dict]) -> float:
    """Token-weighted tau over many generations: sum(new_tokens) / sum(steps)."""
    n = s = 0
    for r in records:
        n += int(r["new_tokens"])
        s += int(r["verify_steps"])
    return n / s if s else float("nan")


def throughput(records: Iterable[Dict], token_key: str = "output_tokens",
               time_key: str = "wall_time") -> float:
    tok = t = 0.0
    for r in records:
        tok += float(r[token_key])
        t += float(r[time_key])
    return tok / t if t > 0 else float("nan")


def speedup(sd_records: Sequence[Dict], ar_records: Sequence[Dict]) -> float:
    return throughput(sd_records) / throughput(ar_records)


def geometric_mean(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs]
    if not xs or any(x <= 0 or math.isnan(x) for x in xs):
        return float("nan")
    return float(math.exp(sum(math.log(x) for x in xs) / len(xs)))


def paired_bootstrap_speedup(sd: Sequence[Dict], ar: Sequence[Dict], key: str = "qid_turn",
                             n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> Dict:
    """Bootstrap CI for throughput ratio, resampling prompts (pairs) with replacement.

    Both lists must describe the same prompts; pairing is by `key`.
    Valid at batch size 1, where each prompt has its own wall time.
    """
    a = {r[key]: r for r in sd}
    b = {r[key]: r for r in ar}
    keys = sorted(set(a) & set(b), key=str)
    if not keys:
        raise ValueError("no paired prompts")
    st = np.array([a[k]["output_tokens"] for k in keys], dtype=float)
    sw = np.array([a[k]["wall_time"] for k in keys], dtype=float)
    at = np.array([b[k]["output_tokens"] for k in keys], dtype=float)
    aw = np.array([b[k]["wall_time"] for k in keys], dtype=float)
    point = (st.sum() / sw.sum()) / (at.sum() / aw.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    ratios = (st[idx].sum(1) / sw[idx].sum(1)) / (at[idx].sum(1) / aw[idx].sum(1))
    lo, hi = np.quantile(ratios, [alpha / 2, 1 - alpha / 2])
    return {"speedup": float(point), "lo": float(lo), "hi": float(hi), "n_pairs": len(keys)}


def overlap_throughput(spans: Sequence[Dict], t0: float, t1: float) -> float:
    """Tokens/s inside [t0, t1], pro-rating each request's tokens over its lifetime.

    spans: dicts with start, end (seconds) and output_tokens.
    """
    if t1 <= t0:
        return float("nan")
    tok = 0.0
    for s in spans:
        dur = s["end"] - s["start"]
        if dur <= 0:
            continue
        ov = max(0.0, min(s["end"], t1) - max(s["start"], t0))
        tok += s["output_tokens"] * ov / dur
    return tok / (t1 - t0)


def full_occupancy_window(spans: Sequence[Dict], concurrency: int):
    """Interval during which `concurrency` requests were simultaneously in flight.

    Returns (t0, t1) = first time active reaches `concurrency` and the last time
    it is still at `concurrency`; None if never reached (e.g., fewer requests than
    the concurrency cap).
    """
    events = []
    for s in spans:
        events.append((s["start"], 1))
        events.append((s["end"], -1))
    # ends before starts at identical timestamps (a slot is freed, then reused)
    events.sort(key=lambda e: (e[0], e[1]))
    active, t0, t1 = 0, None, None
    for t, d in events:
        before = active
        active += d
        if active >= concurrency and t0 is None:
            t0 = t
        if before >= concurrency and active < concurrency:
            t1 = t
    if t0 is None or t1 is None or t1 <= t0:
        return None
    return t0, t1
