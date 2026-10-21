"""Experiment 0b — calibration stage 2: corner-address width and energy.

Stage 1 fixed n_hyperplanes=16, which gives 65k hypercube corners for a 20k
corpus — almost every bucket empty, so ripple entry degenerates to random
nodes. The hypothesis's own "logarithmic layout" suggests the address width
should scale as log2(N / bucket_size). This stage sweeps the address width,
initial energy and the strongest stage-1 physics, and reports the Pareto
frontier of recall vs work.
"""

import itertools

import numpy as np

from _common import save, threadpool_limits
from ripple_search import BruteForce, RippleEngine
from ripple_search.timing import recall_at_k
from ripple_search.workloads import make_corpus, make_queries

K = 10
N = 20_000
D = 64

X, _, centers = make_corpus(N, D, 50, spread=0.30, seed=0)
Q, _ = make_queries(centers, 80, "uniform", spread=0.30, seed=99)
bf = BruteForce()
bf.build(X)
truth = [bf.query(q, k=K) for q in Q]

rows = []
for n_hp in [8, 10, 12, 16]:
    e = RippleEngine(n_neighbors=32, n_hyperplanes=n_hp, seed=0)
    e.build(X)
    for decay, gamma, theta, n_entry, e0 in itertools.product(
            [0.90, 0.95], [1.0, 2.0], [1e-4, 1e-5], [8, 16], [1.0, 4.0]):
        e.decay, e.gamma, e.theta, e.n_entry, e.init_energy = (
            decay, gamma, theta, n_entry, e0)
        rec, vis = [], []
        with threadpool_limits(limits=1):
            for q, t in zip(Q, truth):
                r = e.query(q, k=K, learn=False)
                rec.append(recall_at_k(r, t, K))
                vis.append(e.last_stats.visited)
        rows.append({"n_hyperplanes": n_hp, "decay": decay, "gamma": gamma,
                     "theta": theta, "n_entry": n_entry, "init_energy": e0,
                     "recall@10": float(np.mean(rec)),
                     "mean_visited": float(np.mean(vis)),
                     "visited_frac": float(np.mean(vis) / N)})
        print(rows[-1])

# Pareto frontier: maximize recall, minimize visited.
rows_sorted = sorted(rows, key=lambda r: r["mean_visited"])
pareto, best_r = [], -1.0
for r in rows_sorted:
    if r["recall@10"] > best_r:
        pareto.append(r)
        best_r = r["recall@10"]
feasible = [r for r in rows if r["visited_frac"] <= 0.05]
best = max(feasible or rows, key=lambda r: (r["recall@10"], -r["mean_visited"]))
print("\nPARETO:")
for r in pareto:
    print(f"  visited={r['mean_visited']:.0f} ({r['visited_frac']*100:.1f}%) "
          f"recall={r['recall@10']:.3f}  <- {r}")
print("\nBEST (<=5% visited):", best)
save("exp0b_calibration_stage2", {"description": __doc__, "rows": rows,
                                  "pareto": pareto, "best": best})
