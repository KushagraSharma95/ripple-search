"""Experiment 1b — recall-matched comparison at N=100k.

exp1 holds every engine at fixed parameters while N grows, so at N=100k the
ripple (quality config) sits at recall 0.87 and HNSW at ef=64 sits at 0.64 —
different quality points, not directly comparable. This experiment sweeps
HNSW's ef at N=100k to find its latency AT OR ABOVE the ripple's measured
recall, so the headline comparison is recall-matched rather than asserted.
"""

import json
import time

import numpy as np

from _common import RESULTS, save, threadpool_limits
from ripple_search import BruteForce, HNSW
from ripple_search.timing import recall_at_k, summarize, time_queries
from ripple_search.workloads import make_corpus, make_queries

K = 10
N = 100_000
D = 64

X, _, centers = make_corpus(N, D, 50, spread=0.30, seed=0)
Q, _ = make_queries(centers, 200, "uniform", spread=0.30, seed=1)

bf = BruteForce()
bf.build(X)
truth = [bf.query(q, k=K) for q in Q]

ripple_ref = None
exp1 = json.loads((RESULTS / "exp1_scaling_n.json").read_text())
for row in exp1["rows"]:
    if row["n"] == N:
        ripple_ref = row["engines"]["ripple_quality"]
print("ripple reference @100k:", {k: ripple_ref[k] for k in ("recall@10", "p50_ms")})

hnsw = HNSW(M=16, ef_construction=200, ef=64, seed=0)
t0 = time.perf_counter()
hnsw.build(X)
build_s = time.perf_counter() - t0

rows = []
for ef in [64, 128, 256, 512, 1024]:
    hnsw.index.set_ef(ef)
    with threadpool_limits(limits=1):
        res, lat = time_queries(hnsw.query, Q, k=K)
    rows.append({"ef": ef, **summarize(lat),
                 "recall@10": float(np.mean([recall_at_k(r, t, K) for r, t in zip(res, truth)]))})
    print(rows[-1])

matched = next((r for r in rows if r["recall@10"] >= ripple_ref["recall@10"]), rows[-1])
save("exp1b_recall_matched", {
    "description": __doc__, "n": N, "hnsw_build_s": build_s,
    "ripple_reference": ripple_ref, "hnsw_ef_sweep": rows,
    "hnsw_at_matched_recall": matched,
    "latency_ratio_p50": ripple_ref["p50_ms"] / matched["p50_ms"],
})
print(f"\nAt recall >= {ripple_ref['recall@10']:.3f}: hnsw p50={matched['p50_ms']:.3f}ms "
      f"vs ripple p50={ripple_ref['p50_ms']:.1f}ms "
      f"-> ripple is {ripple_ref['p50_ms']/matched['p50_ms']:.0f}x slower")
