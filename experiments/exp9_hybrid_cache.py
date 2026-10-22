"""Experiment 9 — the second win attempt: one cache policy for all workloads.

exp4 found specialists: ghost-frequency LFU wins zipf, bounded-prefetch LRU
wins drifting, uniform ties. HybridCache combines frequency-gated admission
with budget-bounded semantic prefetch.

Protocol (pre-registered here):
  1. Tune (prefetch_m, prefetch_frac) on SEED 0 only.
  2. Freeze the winner; evaluate on seeds 1 and 2, never re-tuned.
  3. Win condition: on every workload, hybrid's mean-across-capacities hit
     rate >= the best exp4 arm for that workload (within 1 sd), on the
     held-out seeds.

Run with SEED=0 TUNE=1 for the tuning pass; SEED=n for evaluation passes.
"""

import os

import numpy as np

from _common import save, threadpool_limits
from ripple_search import (BruteForce, LFUCache, LFUPlainCache, LRUCache,
                           RippleEngine, ThermoCache)
from ripple_search.hybrid_cache import HybridCache
from ripple_search.workloads import make_corpus, make_queries

K = 10
N = 20_000
D = 64
N_TOPICS = 100
N_STREAM = 4_000
CAPACITIES = [250, 500, 1000, 2000]
SEED = int(os.environ.get("SEED", "0"))
TUNE = os.environ.get("TUNE", "0") == "1"

X, _, centers = make_corpus(N, D, N_TOPICS, spread=0.30, seed=SEED)
bf = BruteForce()
bf.build(X)
rip = RippleEngine(n_neighbors=32, n_hyperplanes=10, decay=0.90, gamma=1.0,
                   theta=1e-4, n_entry=16, init_energy=4.0, seed=SEED)
rip.build(X)

import pickle
from pathlib import Path

_cache = Path(f"/tmp/exp9_streams_seed{SEED}.pkl")
if _cache.exists():
    streams = pickle.loads(_cache.read_bytes())
    print(f"streams loaded from {_cache}")
else:
    streams = {}
    for workload in ["zipf", "drifting", "uniform"]:
        Q, _ = make_queries(centers, N_STREAM, workload, spread=0.30, seed=4 + 10 * SEED)
        needed, neighborhoods = [], []
        with threadpool_limits(limits=1):
            for q in Q:
                needed.append(bf.query(q, k=K))
                rip.query(q, k=K, learn=False)
                # Keep only the top-500 activation energies: the full ~10k-entry
                # dicts per query exhaust RAM across 12k streams, and no policy
                # here admits more than a few hundred prefetch candidates.
                # Every policy sees the identical truncated streams.
                act = rip.last_stats.activated_energy
                top = sorted(act.items(), key=lambda kv: -kv[1])[:500]
                neighborhoods.append(dict(top))
        streams[workload] = (needed, neighborhoods)
    _cache.write_bytes(pickle.dumps(streams))


def run_policy(make_cache):
    """Returns {workload: mean hit rate across capacities}."""
    res = {}
    for workload, (needed, neighborhoods) in streams.items():
        rates = []
        for cap in CAPACITIES:
            c = make_cache(cap)
            for t, (need, nb) in enumerate(zip(needed, neighborhoods)):
                c.probe(need, float(t))
                c.admit(need, nb, float(t))
            rates.append(c.hit_rate)
        res[workload] = float(np.mean(rates))
    return res


out = {"description": __doc__, "seed": SEED, "capacities": CAPACITIES}

if TUNE:
    grid = []
    for w in [0.05, 0.1, 0.2, 0.3, 0.5]:
        for m in [50, 100]:
            for frac in [0.25, 0.5]:
                r = run_policy(lambda cap, w=w, m=m, f=frac: HybridCache(
                    cap, prefetch_m=m, prefetch_frac=f, window_frac=w))
                score = sum(r.values())
                grid.append({"window_frac": w, "prefetch_m": m,
                             "prefetch_frac": frac, **r, "sum": score})
                print(f"w={w} m={m} frac={frac}: zipf={r['zipf']:.4f} "
                      f"drift={r['drifting']:.4f} unif={r['uniform']:.4f} sum={score:.4f}")
    best = max(grid, key=lambda g: g["sum"])
    print("BEST:", best)
    out["tuning_grid"] = grid
    out["best"] = best
    save("exp9_hybrid_tuning", out)
else:
    M = int(os.environ.get("PREFETCH_M", "50"))
    FRAC = float(os.environ.get("PREFETCH_FRAC", "0.25"))
    W = float(os.environ.get("WINDOW_FRAC", "0.1"))
    tau = N_STREAM / 20
    policies = {
        "hybrid": lambda cap: HybridCache(cap, prefetch_m=M, prefetch_frac=FRAC,
                                          window_frac=W),
        "lru": lambda cap: LRUCache(cap, prefetch=False),
        "lfu_plain": lambda cap: LFUPlainCache(cap),
        "lfu_ghost_admission": lambda cap: LFUCache(cap),
        "lru_prefetch_top100": None,  # handled below (needs truncation)
        "thermo_prefetch": lambda cap: ThermoCache(cap, tau=tau, prefetch=True),
    }
    results = {}
    for name, mk in policies.items():
        if name == "lru_prefetch_top100":
            def run_lru_p100():
                res = {}
                for workload, (needed, neighborhoods) in streams.items():
                    rates = []
                    for cap in CAPACITIES:
                        c = LRUCache(cap, prefetch=True)
                        for t, (need, nb) in enumerate(zip(needed, neighborhoods)):
                            c.probe(need, float(t))
                            top = dict(sorted(nb.items(), key=lambda kv: -kv[1])[:100])
                            c.admit(need, top, float(t))
                        rates.append(c.hit_rate)
                    res[workload] = float(np.mean(rates))
                return res
            results[name] = run_lru_p100()
        else:
            results[name] = run_policy(mk)
        print(name, {k: round(v, 4) for k, v in results[name].items()})
    out["hybrid_params"] = {"prefetch_m": M, "prefetch_frac": FRAC}
    out["policies"] = results
    suffix = "" if SEED == 0 else f"_seed{SEED}"
    save(f"exp9_hybrid_cache{suffix}", out)
