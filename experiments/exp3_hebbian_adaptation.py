"""Experiment 3 — CLAIM: "the network organically self-optimizes, burning
highly traveled paths into permanent shortcuts; as more query cycles enter the
system the search becomes more intelligent, faster, and more precise."

Test: identical twin engines (same build, same seed), one adaptive (Hebbian +
shortcut burning ON), one frozen. Stream 4,000 queries from workloads with
exploitable locality (zipf, drifting). Per 250-query window measure work
(nodes visited, distance computations) and recall@10 vs exact truth.

Also the catch the hypothesis never mentions: after adapting, evaluate both
twins on 300 UNIFORM out-of-distribution queries. If adaptation warps the
graph toward the hot topics, OOD recall should reveal the price.
"""

import numpy as np

from _common import save, threadpool_limits
from ripple_search import BruteForce, RippleEngine
from ripple_search.timing import recall_at_k
from ripple_search.workloads import make_corpus, make_queries

K = 10
N = 20_000
D = 64
N_TOPICS = 100
N_STREAM = 4_000
WINDOW = 250

X, _, centers = make_corpus(N, D, N_TOPICS, spread=0.30, seed=0)
bf = BruteForce()
bf.build(X)

out = {"description": __doc__, "n": N, "d": D, "workloads": {}}

for workload in ["zipf", "drifting"]:
    Q, _ = make_queries(centers, N_STREAM, workload, spread=0.30, seed=2)
    truth = [bf.query(q, k=K) for q in Q]

    engines = {}
    for name in ["adaptive", "frozen"]:
        e = RippleEngine(n_neighbors=32, n_hyperplanes=10, decay=0.90, gamma=1.0,
                   theta=1e-4, n_entry=16, init_energy=4.0, seed=0)
        e.build(X)
        if name == "frozen":
            e.freeze()
        engines[name] = e

    windows = {name: [] for name in engines}
    with threadpool_limits(limits=1):
        for name, e in engines.items():
            for w0 in range(0, N_STREAM, WINDOW):
                vis, dc, rec = [], [], []
                for i in range(w0, min(w0 + WINDOW, N_STREAM)):
                    r = e.query(Q[i], k=K, learn=(name == "adaptive"))
                    vis.append(e.last_stats.visited)
                    dc.append(e.last_stats.dist_computations)
                    rec.append(recall_at_k(r, truth[i], K))
                windows[name].append({
                    "window_start": w0,
                    "mean_visited": float(np.mean(vis)),
                    "mean_dist_computations": float(np.mean(dc)),
                    "recall@10": float(np.mean(rec)),
                })

    # Out-of-distribution evaluation after the stream.
    Qo, _ = make_queries(centers, 300, "uniform", spread=0.30, seed=3)
    truth_o = [bf.query(q, k=K) for q in Qo]
    ood = {}
    with threadpool_limits(limits=1):
        for name, e in engines.items():
            rec, vis = [], []
            for q, t in zip(Qo, truth_o):
                r = e.query(q, k=K, learn=False)
                rec.append(recall_at_k(r, t, K))
                vis.append(e.last_stats.visited)
            ood[name] = {"recall@10": float(np.mean(rec)),
                         "mean_visited": float(np.mean(vis))}

    out["workloads"][workload] = {"windows": windows, "ood_after_stream": ood}
    a_first, a_last = windows["adaptive"][0], windows["adaptive"][-1]
    print(f"{workload}: adaptive visited {a_first['mean_visited']:.0f} -> "
          f"{a_last['mean_visited']:.0f}, recall {a_first['recall@10']:.3f} -> "
          f"{a_last['recall@10']:.3f} | OOD adaptive={ood['adaptive']['recall@10']:.3f} "
          f"frozen={ood['frozen']['recall@10']:.3f}")

save("exp3_hebbian_adaptation", out)
