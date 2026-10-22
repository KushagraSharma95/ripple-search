"""Experiment 6 — CLAIM: "as data ages the system merges old nodes into dense
historical macro-nodes; time effectively self-indexes the database, keeping
the active search field perpetually small and lightning-fast" — at what cost
to recall on archival queries?

Test: N=50k docs with timestamps over 1000 days (90% archive, 10% recent).
Queries: a recent-leaning stream (80% target recent docs) plus a pure archival
stream. Measure visited count, latency, and recall@10 vs exact truth before
and after consolidation, separately for recent-targeting and archive-targeting
queries.
"""

import time

import numpy as np

from _common import save, threadpool_limits
from ripple_search import BruteForce, RippleEngine
from ripple_search.timing import recall_at_k
from ripple_search.workloads import make_corpus, make_queries

K = 10
N = 50_000
D = 64
N_TOPICS = 100
NOW = 1000.0

rng = np.random.default_rng(6)
X, topics, centers = make_corpus(N, D, N_TOPICS, spread=0.30, seed=0)
# 90% archival (age uniform 100..1000 days), 10% recent (age 0..100 days)
ts = np.where(rng.random(N) < 0.9,
              NOW - rng.uniform(100, 1000, N),
              NOW - rng.uniform(0, 100, N))

bf = BruteForce()
bf.build(X)

# Recent-targeting queries: perturbations of RECENT documents; archival ones
# of OLD documents — so each stream genuinely needs its era's nodes.
recent_ids = np.where(NOW - ts <= 100)[0]
old_ids = np.where(NOW - ts > 500)[0]


def doc_queries(ids, n, seed):
    r = np.random.default_rng(seed)
    pick = r.choice(ids, size=n)
    Q = X[pick] + 0.15 * r.standard_normal((n, D)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return Q.astype(np.float32)


Q_recent = doc_queries(recent_ids, 200, seed=11)
Q_old = doc_queries(old_ids, 200, seed=12)
truth_recent = [bf.query(q, k=K) for q in Q_recent]
truth_old = [bf.query(q, k=K) for q in Q_old]

# Ground truth is over the FULL corpus, which is 90% archival — so even
# "recent-seeded" queries have mostly-archival truth sets. Published so the
# recall numbers cannot be read as "recall on recent data".
recent_set = set(recent_ids.tolist())
def _recent_frac(truths):
    return float(np.mean([[i in recent_set for i in t] for t in truths]))
truth_composition = {"recent_seeded_truth_recent_fraction": _recent_frac(truth_recent),
                     "archive_seeded_truth_recent_fraction": _recent_frac(truth_old)}


def evaluate(engine, Q, truth):
    rec, vis, dc, lat = [], [], [], []
    with threadpool_limits(limits=1):
        for q, t in zip(Q, truth):
            t0 = time.perf_counter()
            r = engine.query(q, k=K, learn=False)
            lat.append(time.perf_counter() - t0)
            rec.append(recall_at_k(r, t, K))
            vis.append(engine.last_stats.visited)
            # The honest work metric: true distance computations, INCLUDING
            # macro-member expansion after consolidation (visited alone counts
            # only graph settles and understates post-consolidation work).
            dc.append(engine.last_stats.dist_computations)
    return {"recall@10": float(np.mean(rec)),
            "mean_visited": float(np.mean(vis)),
            "mean_dist_computations": float(np.mean(dc)),
            "p50_ms": float(np.percentile(lat, 50) * 1e3)}


rip = RippleEngine(n_neighbors=32, n_hyperplanes=10, decay=0.90, gamma=1.0,
                   theta=1e-4, n_entry=16, init_energy=4.0, seed=0)
rip.build(X, timestamps=ts)

before = {"recent_queries": evaluate(rip, Q_recent, truth_recent),
          "old_queries": evaluate(rip, Q_old, truth_old)}

n_macro = rip.consolidate(now=NOW, base_window=100.0, max_group=32)
active = int(rip.active_mask[: N].sum())

after = {"recent_queries": evaluate(rip, Q_recent, truth_recent),
         "old_queries": evaluate(rip, Q_old, truth_old)}

out = {"description": __doc__, "n": N, "macro_nodes_created": n_macro,
       "active_original_nodes_after": active,
       "compression_ratio": float(N / max(active + n_macro, 1)),
       "truth_composition": truth_composition,
       "before": before, "after": after}
print("macro nodes:", n_macro, "| active originals:", active)
print("truth composition:", truth_composition)
for era in ["recent_queries", "old_queries"]:
    print(f"{era}: recall {before[era]['recall@10']:.3f} -> {after[era]['recall@10']:.3f}, "
          f"visited {before[era]['mean_visited']:.0f} -> {after[era]['mean_visited']:.0f}, "
          f"dists {before[era]['mean_dist_computations']:.0f} -> {after[era]['mean_dist_computations']:.0f}, "
          f"p50 {before[era]['p50_ms']:.2f} -> {after[era]['p50_ms']:.2f}ms")

save("exp6_temporal_consolidation", out)
