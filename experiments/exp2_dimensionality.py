"""Experiment 2 — CLAIM: "the higher the number of dimensions, the sharper
the pruning: search becomes MORE accurate because geometric pruning angles get
sharper; a single corner check rules out a massive percentage of the database."

Test: fix N=20k, sweep D from 2 to 512. Measure:
  - VP-tree distance computations as a fraction of N (triangle-inequality
    pruning effectiveness — the claim says this should FALL with D)
  - relative contrast (d_max-d_min)/d_min (distance concentration)
  - ripple recall@10 and visited count at FIXED energy parameters
  - HNSW recall at fixed ef (graceful-degradation reference)
  - Hamming separation of random hypercube codes (the hyperdimensional-
    computing counterpoint: for random SYMBOLIC codes, high-D does help)

Run on clustered data (realistic) AND i.i.d. Gaussian (the worst case the
curse-of-dimensionality literature describes).
"""

import numpy as np

from _common import save, threadpool_limits
from ripple_search import BruteForce, HNSW, RippleEngine, VPTree
from ripple_search.metric_tree import hamming_separation, relative_contrast
from ripple_search.timing import recall_at_k, time_queries
from ripple_search.workloads import make_corpus, make_queries

K = 10
N = 20_000
N_QUERIES = 100
DIMS = [2, 4, 8, 16, 32, 64, 128, 256, 512]

rng = np.random.default_rng(7)
rows = []
for regime in ["clustered", "iid_gaussian"]:
    for d in DIMS:
        if regime == "clustered":
            X, _, centers = make_corpus(N, d, 50, spread=0.30, seed=0)
            Q, _ = make_queries(centers, N_QUERIES, "uniform", spread=0.30, seed=1)
        else:
            X = rng.standard_normal((N, d)).astype(np.float32)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            Q = rng.standard_normal((N_QUERIES, d)).astype(np.float32)
            Q /= np.linalg.norm(Q, axis=1, keepdims=True)

        bf = BruteForce()
        bf.build(X)
        truth = [bf.query(q, k=K) for q in Q]

        vp = VPTree(X, seed=0)
        counts, vres = [], []
        for q in Q:
            r, c = vp.query(q, k=K)
            vres.append(r)
            counts.append(c)

        rip = RippleEngine(n_neighbors=32, n_hyperplanes=10, decay=0.90, gamma=1.0,
                   theta=1e-4, n_entry=16, init_energy=4.0, seed=0)
        rip.build(X)
        visited, rres = [], []
        with threadpool_limits(limits=1):
            for q in Q:
                rres.append(rip.query(q, k=K, learn=False))
                visited.append(rip.last_stats.visited)

        hnsw = HNSW(M=16, ef_construction=200, ef=64, seed=0)
        hnsw.build(X)
        with threadpool_limits(limits=1):
            hres, _ = time_queries(hnsw.query, Q, warmup=5, k=K)

        ham_mean, ham_std = hamming_separation(N, d, rng)
        row = {
            "regime": regime,
            "d": d,
            "vp_tree_dist_fraction": float(np.mean(counts) / N),
            "vp_tree_recall@10": float(np.mean([recall_at_k(r, t, K) for r, t in zip(vres, truth)])),
            "relative_contrast": relative_contrast(X[:5000], Q[:20]),
            "ripple_recall@10": float(np.mean([recall_at_k(r, t, K) for r, t in zip(rres, truth)])),
            "ripple_mean_visited": float(np.mean(visited)),
            "hnsw_recall@10": float(np.mean([recall_at_k(r, t, K) for r, t in zip(hres, truth)])),
            "hamming_sep_mean_frac": ham_mean,
            "hamming_sep_std_frac": ham_std,
        }
        rows.append(row)
        print(f"{regime} d={d}: vp_frac={row['vp_tree_dist_fraction']:.3f} "
              f"contrast={row['relative_contrast']:.2f} "
              f"ripple_recall={row['ripple_recall@10']:.3f} "
              f"hnsw_recall={row['hnsw_recall@10']:.3f}")

save("exp2_dimensionality", {"description": __doc__, "n": N, "k": K, "rows": rows})
