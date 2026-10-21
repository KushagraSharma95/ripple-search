"""Benchmark harness: warmup, GC control, percentile reporting, recall.

Discipline (per ann-benchmarks methodology):
  - report recall together with latency, never latency alone
  - separate index BUILD time from QUERY time
  - warm up before measuring; disable GC inside the timed region
  - report p50/p95/mean over many queries, single-threaded
"""

from __future__ import annotations

import gc
import time

import numpy as np


def recall_at_k(result: list[int], truth: list[int], k: int) -> float:
    return len(set(result[:k]) & set(truth[:k])) / k


def time_queries(fn, queries, warmup: int = 20, **kw):
    """fn(q, **kw) -> result list. Returns (results, latencies_seconds)."""
    for q in queries[: min(warmup, len(queries))]:
        fn(q, **kw)
    results, lat = [], []
    gc_was = gc.isenabled()
    gc.disable()
    try:
        for q in queries:
            t0 = time.perf_counter()
            r = fn(q, **kw)
            lat.append(time.perf_counter() - t0)
            results.append(r)
    finally:
        if gc_was:
            gc.enable()
    return results, np.array(lat)


def summarize(lat: np.ndarray) -> dict:
    return {
        "mean_ms": float(lat.mean() * 1e3),
        "p50_ms": float(np.percentile(lat, 50) * 1e3),
        "p95_ms": float(np.percentile(lat, 95) * 1e3),
        "qps_single_thread": float(1.0 / lat.mean()),
    }
