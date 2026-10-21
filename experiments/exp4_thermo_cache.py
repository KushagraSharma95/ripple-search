"""Experiment 4 — CLAIM: thermodynamic caching ("pump the whole activated
neighborhood into cache with energy-proportional temperature, evict by natural
dissipation") beats reactive LRU/LFU.

Test: generate query streams; for each query the "needed set" is the exact
top-10 documents. Probe the cache for the needed set (hits counted BEFORE
admission), then admit this query's results. Grid of policies ablating the two
bundled mechanisms:

    LRU            classic, exact items only
    LFU            classic, exact items only
    LRU+prefetch   classic eviction, neighborhood admission
    Thermo-exact   temperature eviction, exact items only
    Thermo+prefix  temperature eviction, neighborhood admission (the claim)

Workloads: zipf (locality), drifting (the hypothesis's "train of thought"),
uniform (no locality — where prefetch should pollute and HURT).
"""

import numpy as np

from _common import save, threadpool_limits
from ripple_search import BruteForce, LFUCache, LRUCache, RippleEngine, ThermoCache
from ripple_search.workloads import make_corpus, make_queries

K = 10
N = 20_000
D = 64
N_TOPICS = 100
N_STREAM = 4_000
CAPACITIES = [250, 500, 1000, 2000]

X, _, centers = make_corpus(N, D, N_TOPICS, spread=0.30, seed=0)
bf = BruteForce()
bf.build(X)

rip = RippleEngine(n_neighbors=32, n_hyperplanes=10, decay=0.90, gamma=1.0,
                   theta=1e-4, n_entry=16, init_energy=4.0, seed=0)
rip.build(X)

out = {"description": __doc__, "n": N, "capacities": CAPACITIES, "workloads": {}}

for workload in ["zipf", "drifting", "uniform"]:
    Q, _ = make_queries(centers, N_STREAM, workload, spread=0.30, seed=4)
    # Precompute per-query needed sets and activated neighborhoods once.
    needed, neighborhoods = [], []
    with threadpool_limits(limits=1):
        for q in Q:
            needed.append(bf.query(q, k=K))
            rip.query(q, k=K, learn=False)
            # Energy the ripple deposited across the activated zone.
            neighborhoods.append(dict(rip.last_stats.activated_energy))

    rows = []
    for cap in CAPACITIES:
        tau = N_STREAM / 20  # logical-time cooling constant
        policies = {
            "lru": LRUCache(cap, prefetch=False),
            "lfu": LFUCache(cap),
            "lru_prefetch": LRUCache(cap, prefetch=True),
            "thermo_exact": ThermoCache(cap, tau=tau, prefetch=False),
            "thermo_prefetch": ThermoCache(cap, tau=tau, prefetch=True),
        }
        for name, cache in policies.items():
            for t, (need, nb) in enumerate(zip(needed, neighborhoods)):
                cache.probe(need, float(t))
                cache.admit(need, nb, float(t))
        rows.append({"capacity": cap,
                     **{name: c.hit_rate for name, c in policies.items()}})
        print(f"{workload} cap={cap}: " +
              " ".join(f"{n}={c.hit_rate:.3f}" for n, c in policies.items()))
    out["workloads"][workload] = rows

save("exp4_thermo_cache", out)
