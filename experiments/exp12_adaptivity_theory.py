"""Experiment 12 — when does per-query adaptivity pay? A controlled test of
the difficulty-variance hypothesis.

exp10 measured the adaptive controller's dividend growing with the declared
target (3-13% at 0.90 -> 39-55% at 0.99 on Reuters) and vanishing on a
homogeneous synthetic workload. Hypothesis: the dividend is driven by
QUERY-DIFFICULTY VARIANCE — a fixed budget must be sized for the hardest
queries, so the harvestable waste is the easy-query mass; with zero
variance there is nothing to harvest at any target.

Design: one corpus (synthetic, N=100k, D=64 — fixed substrate), five query
difficulty distributions with (approximately) MATCHED MEAN eps and
increasing spread:

    const    : eps = 0.080 for every query          (zero variance)
    narrow   : eps ~ U[0.06, 0.10]
    medium   : eps ~ U[0.03, 0.13]
    wide     : eps ~ U[0.01, 0.15]
    bimodal  : eps in {0.02, 0.14}, 50/50           (max spread at this mean)

For each distribution: train/val/test = 600/300/300 (fresh seed 419, never
used), ridge controller as in exp10, tau calibrated on val per target
T in {0.90, 0.95, 0.99}, fixed-ef baseline selected on val on the x1.2
grid, both evaluated once on test. The measured quantity is the DIVIDEND:
1 - (controller mean dists / fixed-baseline mean dists), per (distribution,
target) cell, with contract-met flags. Prediction registered up front:
dividend increases with spread at every target, and is ~0 (or negative)
for const at all targets. Published win or lose.
"""

import numpy as np

from _common import save, threadpool_limits
from ripple_search import BruteForce
from ripple_search.beam_search import InstrumentedBeamSearch
from ripple_search.timing import recall_at_k
from ripple_search.workloads import make_corpus

K = 10
TARGETS = [0.90, 0.95, 0.99]
CHECK_EVERY = 8
N_TRAIN, N_VAL, N_TEST = 600, 300, 300
QSEED = 419
CEILING = 768

DISTS = {
    "const": lambda rng, n: np.full(n, 0.080, dtype=np.float32),
    "narrow": lambda rng, n: rng.uniform(0.06, 0.10, n).astype(np.float32),
    "medium": lambda rng, n: rng.uniform(0.03, 0.13, n).astype(np.float32),
    "wide": lambda rng, n: rng.uniform(0.01, 0.15, n).astype(np.float32),
    "bimodal": lambda rng, n: rng.choice([0.02, 0.14], n).astype(np.float32),
}


def expand(feats):
    f = np.asarray(feats, dtype=np.float64)
    e, v, stall, imp, top, kth, ratio, gap, frac = f
    return np.array([
        e, v, stall, imp, top, kth, ratio, gap, frac,
        np.log1p(e), np.log1p(stall), stall / max(e, 1.0),
        float(gap < 0), gap * frac, ratio * ratio,
    ])


X, _, _ = make_corpus(100_000, 64, 50, spread=0.30, seed=0)
bs = InstrumentedBeamSearch(n_neighbors=32, n_hyperplanes=10, n_entry=4, seed=0)
bs.build(X)
bf = BruteForce()
bf.build(X)
bs.X = bf.X

out = {"description": __doc__, "k": K, "query_seed": QSEED, "targets": TARGETS,
       "distributions": {}}

for dname, eps_fn in DISTS.items():
    print(f"=== {dname} ===", flush=True)
    rng = np.random.default_rng(QSEED)
    n = N_TRAIN + N_VAL + N_TEST
    ids = rng.choice(len(bf.X), size=n, replace=False)
    eps = eps_fn(rng, n)
    Q = bf.X[ids] + eps[:, None] * rng.standard_normal((n, bf.X.shape[1])).astype(np.float32)
    Q = (Q / np.linalg.norm(Q, axis=1, keepdims=True)).astype(np.float32)
    truth = []
    for q, sid in zip(Q, ids):
        r = bf.query(q, k=K + 1)
        truth.append([i for i in r if i != sid][:K])
    sl_tr, sl_val, sl_te = (slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_VAL),
                            slice(N_TRAIN + N_VAL, None))

    rows_X, rows_y = [], []
    with threadpool_limits(limits=1):
        for q, t, sid in zip(Q[sl_tr], truth[sl_tr], ids[sl_tr]):
            recs = []

            def cb(feats, snap, recs=recs):
                recs.append((feats, snap()))
                return False

            bs.search(q, k=K + 1, ef=CEILING, check_every=CHECK_EVERY, on_checkpoint=cb)
            for feats, iids in recs:
                rows_X.append(expand(feats))
                rows_y.append(recall_at_k([i for i in iids if i != sid][:K], t, K))
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Xf, yf = np.vstack(rows_X), np.asarray(rows_y)
    scaler = StandardScaler().fit(Xf)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(Xf), yf)
    w = (ridge.coef_ / scaler.scale_).astype(np.float64)
    b = float(ridge.intercept_ - np.dot(ridge.coef_, scaler.mean_ / scaler.scale_))

    def run_pred(sl, tau):
        recs, dists = [], []
        with threadpool_limits(limits=1):
            for q, t, sid in zip(Q[sl], truth[sl], ids[sl]):
                def cb(feats, snap):
                    return float(np.dot(w, expand(feats)) + b) >= tau

                iids, st = bs.search(q, k=K + 1, ef=CEILING,
                                     check_every=CHECK_EVERY, on_checkpoint=cb)
                recs.append(recall_at_k([i for i in iids if i != sid][:K], t, K))
                dists.append(st["dist_computations"])
        return float(np.mean(recs)), float(np.mean(dists))

    def run_fixed(sl, ef):
        recs, dists = [], []
        with threadpool_limits(limits=1):
            for q, t, sid in zip(Q[sl], truth[sl], ids[sl]):
                r = bs.query(q, k=K + 1, mode="fixed", ef=ef)
                recs.append(recall_at_k([i for i in r if i != sid][:K], t, K))
                dists.append(bs.last_dist_count)
        return float(np.mean(recs)), float(np.mean(dists))

    FIXED_GRID = [16, 20, 24, 29, 35, 42, 50, 60, 72, 86, 104, 124, 149,
                  179, 215, 258, 310, 372, 446, 535, 642, 768]
    fixed_val = {ef: run_fixed(sl_val, ef) for ef in FIXED_GRID}
    ceil_recall = fixed_val[CEILING][0] if CEILING in fixed_val else run_fixed(sl_val, CEILING)[0]

    cells = []
    for T in TARGETS:
        if T > ceil_recall:
            cells.append({"target": T, "unachievable": True,
                          "ceiling_recall_val": ceil_recall})
            print(f" T={T}: UNACHIEVABLE (ceiling {ceil_recall:.4f})", flush=True)
            continue
        lo, hi = T - 0.05, 1.10
        best = None
        for _ in range(8):
            mid = (lo + hi) / 2
            rec, dist = run_pred(sl_val, mid)
            if rec >= T:
                best = mid
                hi = mid
            else:
                lo = mid
        tau = best if best is not None else hi
        ok = [ef for ef in FIXED_GRID if fixed_val[ef][0] >= T]
        if not ok:
            cells.append({"target": T, "unachievable": True,
                          "ceiling_recall_val": ceil_recall})
            continue
        ef0 = min(ok, key=lambda e: fixed_val[e][1])
        p_rec, p_dist = run_pred(sl_te, tau)
        f_rec, f_dist = run_fixed(sl_te, ef0)
        dividend = 1 - p_dist / f_dist
        cells.append({"target": T, "tau": tau, "fixed_ef": ef0,
                      "pred_recall": p_rec, "pred_dists": p_dist,
                      "fixed_recall": f_rec, "fixed_dists": f_dist,
                      "contract_met": bool(p_rec >= T),
                      "dividend": dividend})
        print(f" T={T}: pred {p_rec:.4f}@{p_dist:.0f} vs fixed(ef={ef0}) "
              f"{f_rec:.4f}@{f_dist:.0f} -> dividend {dividend*100:+.1f}%", flush=True)

    out["distributions"][dname] = {
        "eps_mean": float(np.mean(eps)), "eps_std": float(np.std(eps)),
        "ceiling_recall_val": ceil_recall, "cells": cells}
    save("exp12_adaptivity_theory", out)
