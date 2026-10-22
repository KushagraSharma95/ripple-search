"""Experiment 4 — CLAIM: thermodynamic caching ("pump the whole activated
neighborhood into cache with energy-proportional temperature, evict by natural
dissipation") beats reactive LRU/LFU.

Test: generate query streams; for each query the "needed set" is the exact
top-10 documents. Probe the cache for the needed set (hits counted BEFORE
admission), then admit this query's results. Grid of policies ablating the two
bundled mechanisms:

    lru                   classic LRU, exact items only
    lfu_plain             textbook in-cache LFU, exact items only
    lfu_ghost_admission   LFU with ghost counts + gated admission (TinyLFU-family)
    lru_prefetch_all      LRU eviction, FULL activated-neighborhood admission
    lru_prefetch_top100   LRU eviction, top-100-energy bounded admission
    thermo_exact          temperature eviction, exact items only
    thermo_prefetch       temperature eviction, FULL neighborhood (the claim)
    thermo_prefetch_top100  temperature eviction, bounded admission

Workloads: zipf (locality), drifting (the hypothesis's "train of thought"),
uniform (no locality — where prefetch should pollute and HURT).
"""

import os

import numpy as np

from _common import save, threadpool_limits
from ripple_search import (BruteForce, LFUCache, LFUPlainCache, LRUCache,
                           RippleEngine, ThermoCache)
from ripple_search.workloads import make_corpus, make_queries

K = 10
N = 20_000
D = 64
N_TOPICS = 100
N_STREAM = 4_000
CAPACITIES = [250, 500, 1000, 2000]
SEED = int(os.environ.get("SEED", "0"))

X, _, centers = make_corpus(N, D, N_TOPICS, spread=0.30, seed=SEED)
bf = BruteForce()
bf.build(X)

rip = RippleEngine(n_neighbors=32, n_hyperplanes=10, decay=0.90, gamma=1.0,
                   theta=1e-4, n_entry=16, init_energy=4.0, seed=SEED)
rip.build(X)

out = {"description": __doc__, "n": N, "capacities": CAPACITIES, "seed": SEED,
       "workloads": {}}

for workload in ["zipf", "drifting", "uniform"]:
    Q, _ = make_queries(centers, N_STREAM, workload, spread=0.30, seed=4 + 10 * SEED)
    # Precompute per-query needed sets and activated neighborhoods once.
    needed, neighborhoods = [], []
    with threadpool_limits(limits=1):
        for q in Q:
            needed.append(bf.query(q, k=K))
            rip.query(q, k=K, learn=False)
            # Energy the ripple deposited across the activated zone.
            neighborhoods.append(dict(rip.last_stats.activated_energy))

    def top_energy(nb: dict, m: int = 100) -> dict:
        """Charitable bounded prefetch: only the m hottest neighborhood items."""
        return dict(sorted(nb.items(), key=lambda kv: -kv[1])[:m])

    rows = []
    for cap in CAPACITIES:
        tau = N_STREAM / 20  # logical-time cooling constant
        policies = {
            "lru": LRUCache(cap, prefetch=False),
            "lfu_plain": LFUPlainCache(cap),
            "lfu_ghost_admission": LFUCache(cap),
            "lru_prefetch_all": LRUCache(cap, prefetch=True),
            "lru_prefetch_top100": LRUCache(cap, prefetch=True),
            "thermo_exact": ThermoCache(cap, tau=tau, prefetch=False),
            "thermo_prefetch": ThermoCache(cap, tau=tau, prefetch=True),
            "thermo_prefetch_top100": ThermoCache(cap, tau=tau, prefetch=True),
        }
        for name, cache in policies.items():
            for t, (need, nb) in enumerate(zip(needed, neighborhoods)):
                cache.probe(need, float(t))
                nb_pol = top_energy(nb) if name.endswith("_top100") else nb
                cache.admit(need, nb_pol, float(t))
        rows.append({"capacity": cap,
                     **{name: c.hit_rate for name, c in policies.items()}})
        print(f"{workload} cap={cap}: " +
              " ".join(f"{n}={c.hit_rate:.3f}" for n, c in policies.items()))
    # Mean hit rate across capacities per policy (guards against
    # single-capacity cherry-picking when verdicts are written).
    means = {name: float(np.mean([r[name] for r in rows]))
             for name in rows[0] if name != "capacity"}
    out["workloads"][workload] = {"by_capacity": rows,
                                  "mean_across_capacities": means}

suffix = "" if SEED == 0 else f"_seed{SEED}"
save(f"exp4_thermo_cache{suffix}", out)
