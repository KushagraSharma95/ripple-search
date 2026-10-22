"""Experiment 1 — CLAIM: "As data grows, search speed stays flat because
energy decay restricts the radius" (while traditional engines slow down
"linearly or quadratically").

Test: fix D=64, sweep N from 2k to 100k. Measure per engine: build time,
query latency (p50/p95, single-threaded), recall@10 vs exact ground truth,
and language-independent work counters. If the claim holds, ripple latency
stays flat AND recall stays high as N grows.
"""

import time

import numpy as np

from _common import save, threadpool_limits
from ripple_search import BruteForce, HNSW, RippleEngine, VPTree
from ripple_search.timing import recall_at_k, summarize, time_queries
from ripple_search.workloads import make_corpus, make_queries

K = 10
N_QUERIES = 200
D = 64
N_TOPICS = 50
SIZES = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000]

rows = []
for n in SIZES:
    X, _, centers = make_corpus(n, D, N_TOPICS, spread=0.30, seed=0)
    Q, _ = make_queries(centers, N_QUERIES, "uniform", spread=0.30, seed=1)
    row = {"n": n, "d": D, "engines": {}}

    bf = BruteForce()
    t0 = time.perf_counter()
    bf.build(X)
    bf_build = time.perf_counter() - t0
    with threadpool_limits(limits=1):
        truth, lat = time_queries(bf.query, Q, k=K)
    row["engines"]["brute_force"] = {
        "build_s": bf_build, **summarize(lat), "recall@10": 1.0,
        "work_per_query": n,
    }

    # Two calibrated operating points from exp0/exp0b (Pareto endpoints):
    # budget  = best recall subject to visiting <=5% of the corpus
    # quality = ~0.97 recall at N=20k validation, whatever work it takes
    RIPPLE_CONFIGS = {
        "ripple_budget": dict(n_hyperplanes=8, decay=0.95, gamma=2.0,
                              theta=1e-4, n_entry=8, init_energy=4.0),
        "ripple_quality": dict(n_hyperplanes=10, decay=0.90, gamma=1.0,
                               theta=1e-4, n_entry=16, init_energy=4.0),
    }
    for cfg_name, cfg in RIPPLE_CONFIGS.items():
        rip = RippleEngine(n_neighbors=32, seed=0, **cfg)
        t0 = time.perf_counter()
        rip.build(X)
        rip_build = time.perf_counter() - t0
        visited, dists = [], []

        def rip_query(q, k=K):
            r = rip.query(q, k=k, learn=False)
            visited.append(rip.last_stats.visited)
            dists.append(rip.last_stats.dist_computations)
            return r

        with threadpool_limits(limits=1):
            res, lat = time_queries(rip_query, Q, k=K)
        # The closure also fired during warmup; counters cover timed calls only.
        visited, dists = visited[-len(Q):], dists[-len(Q):]
        row["engines"][cfg_name] = {
            "build_s": rip_build, **summarize(lat),
            "recall@10": float(np.mean([recall_at_k(r, t, K) for r, t in zip(res, truth)])),
            "mean_visited": float(np.mean(visited)),
            "mean_dist_computations": float(np.mean(dists)),
            "visited_fraction_of_n": float(np.mean(visited) / n),
        }

    hnsw = HNSW(M=16, ef_construction=200, ef=64, seed=0)
    t0 = time.perf_counter()
    hnsw.build(X)
    h_build = time.perf_counter() - t0
    with threadpool_limits(limits=1):
        res, lat = time_queries(hnsw.query, Q, k=K)
    row["engines"]["hnsw"] = {
        "build_s": h_build, **summarize(lat),
        "recall@10": float(np.mean([recall_at_k(r, t, K) for r, t in zip(res, truth)])),
    }

    if n <= 20_000:  # python recursion-heavy; trend is established by 20k
        t0 = time.perf_counter()
        vp = VPTree(X, seed=0)
        vp_build = time.perf_counter() - t0
        counts = []

        def vp_query(q, k=K):
            r, c = vp.query(q, k=k)
            counts.append(c)
            return r

        with threadpool_limits(limits=1):
            res, lat = time_queries(vp_query, Q, k=K)
        counts = counts[-len(Q):]
        row["engines"]["vp_tree"] = {
            "build_s": vp_build, **summarize(lat),
            "recall@10": float(np.mean([recall_at_k(r, t, K) for r, t in zip(res, truth)])),
            "mean_dist_computations": float(np.mean(counts)),
            "dist_fraction_of_n": float(np.mean(counts) / n),
        }

    rows.append(row)
    rq = row["engines"]["ripple_quality"]
    print(f"n={n}: ripple_q p50={rq['p50_ms']:.2f}ms recall={rq['recall@10']:.3f} "
          f"visited={rq['mean_visited']:.0f} | "
          f"hnsw p50={row['engines']['hnsw']['p50_ms']:.3f}ms "
          f"recall={row['engines']['hnsw']['recall@10']:.3f}")

save("exp1_scaling_n", {"description": __doc__, "k": K, "n_queries": N_QUERIES,
                        "rows": rows})
