"""Experiment 0 — fair calibration of the ripple engine.

HNSW is run at a sensible tuned operating point (ef=64), so the ripple engine
deserves the same: a grid search over its physics parameters on a validation
corpus (N=20k, D=64, separate query seed from all other experiments). We
report the full recall/work Pareto frontier and pick the configuration that
maximizes recall@10 subject to visiting <= 5% of the corpus per query. All
other experiments use the winning configuration.
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
Q, _ = make_queries(centers, 80, "uniform", spread=0.30, seed=99)  # calibration-only seed
bf = BruteForce()
bf.build(X)
truth = [bf.query(q, k=K) for q in Q]

grid = {
    "n_neighbors": [16, 32],
    "decay": [0.80, 0.90, 0.95],
    "gamma": [1.0, 2.0],
    "theta": [1e-4, 1e-5],
    "n_entry": [8, 16],
}

rows = []
built = {}
for nn in grid["n_neighbors"]:
    e = RippleEngine(n_neighbors=nn, n_hyperplanes=16, seed=0)
    e.build(X)
    built[nn] = e

for nn, decay, gamma, theta, n_entry in itertools.product(*grid.values()):
    e = built[nn]
    e.decay, e.gamma, e.theta, e.n_entry = decay, gamma, theta, n_entry
    rec, vis = [], []
    with threadpool_limits(limits=1):
        for q, t in zip(Q, truth):
            r = e.query(q, k=K, learn=False)
            rec.append(recall_at_k(r, t, K))
            vis.append(e.last_stats.visited)
    rows.append({"n_neighbors": nn, "decay": decay, "gamma": gamma,
                 "theta": theta, "n_entry": n_entry,
                 "recall@10": float(np.mean(rec)),
                 "mean_visited": float(np.mean(vis)),
                 "visited_frac": float(np.mean(vis) / N)})
    print(rows[-1])

feasible = [r for r in rows if r["visited_frac"] <= 0.05]
best = max(feasible or rows, key=lambda r: (r["recall@10"], -r["mean_visited"]))
print("\nBEST:", best)
save("exp0_calibration", {"description": __doc__, "grid": {k: v for k, v in grid.items()},
                          "rows": rows, "best": best})
