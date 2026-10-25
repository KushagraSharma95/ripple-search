"""Experiment 10 — learned declared-recall termination (the DARTH-protocol attack).

A small regressor predicts the search's CURRENT recall from cheap dynamics
features at checkpoints; the search stops when the predicted recall clears
the declared target. Protocol:

  - queries: 1,200 per dataset with heterogeneous difficulty (per-query
    noise eps ~ U[0.05, 0.5] around corpus vectors; query vector excluded
    from its own ground truth), split 600 train / 300 val / 300 test with
    seeds disjoint from all prior experiments
  - training: checkpoint feature vectors labeled with true running recall,
    collected at the ef ceiling on TRAIN queries only
  - model: ridge regression on expanded features (deployment = one dot
    product per checkpoint); a gradient-boosted reference model is fit to
    measure how much headroom the linear deployment leaves
  - calibration: per declared target T in {0.90, 0.95, 0.99}, the stop
    threshold tau_T is the smallest value whose achieved mean recall on VAL
    clears T; frozen before touching TEST
  - test: achieved recall + distance computations on TEST, against the two
    baselines on identical queries: the cheapest fixed ef that clears T,
    and the cheapest (ef, patience) heuristic early-exit that clears T

Win condition: at each declared target, achieved recall >= T with fewer
mean distance computations than both baselines.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from _common import ROOT, save, threadpool_limits
from ripple_search import BruteForce
from ripple_search.beam_search import BeamSearch, InstrumentedBeamSearch
from ripple_search.timing import recall_at_k

K = 10
TARGETS = [0.90, 0.95, 0.99]
CHECK_EVERY = 8
N_TRAIN, N_VAL, N_TEST = 600, 300, 300
GLOVE_DIR = Path("/home/user/datasets/glove")


def hetero_queries(X, n, seed):
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(X), size=n, replace=False)
    eps = rng.uniform(0.02, 0.25, size=n).astype(np.float32)
    Q = X[ids] + eps[:, None] * rng.standard_normal((n, X.shape[1])).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return Q.astype(np.float32), ids


def truth_excluding_self(bf, Q, self_ids):
    out = []
    for q, sid in zip(Q, self_ids):
        r = bf.query(q, k=K + 1)
        out.append([i for i in r if i != sid][:K])
    return out


def expand(feats):
    f = np.asarray(feats, dtype=np.float64)
    e, v, stall, imp, top, kth, ratio, gap, frac = f
    return np.array([
        e, v, stall, imp, top, kth, ratio, gap, frac,
        np.log1p(e), np.log1p(stall), stall / max(e, 1.0),
        float(gap < 0), gap * frac, ratio * ratio,
    ])


def load_dataset(name):
    if name == "glove100":
        X = np.load(GLOVE_DIR / "glove100.npy")
        nbr = np.load(GLOVE_DIR / "glove100_knn32.npy")
        bs = InstrumentedBeamSearch(n_neighbors=32, n_hyperplanes=12, n_entry=4, seed=0)
        bs.load_graph(X, nbr)
        ceiling = 768
    elif name == "reuters_lsa256":
        import nltk

        nltk.data.path.insert(0, str(ROOT.parent / "nltk_data"))
        from nltk.corpus import reuters
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [reuters.raw(f) for f in sorted(reuters.fileids())]
        vec = TfidfVectorizer(stop_words="english", min_df=3, max_features=50_000,
                              sublinear_tf=True)
        svd = TruncatedSVD(n_components=256, random_state=0)
        X = svd.fit_transform(vec.fit_transform(texts)).astype(np.float32)
        X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
        bs = InstrumentedBeamSearch(n_neighbors=32, n_hyperplanes=10, n_entry=4, seed=0)
        bs.build(X)
        ceiling = 768
    else:  # synthetic_hetero
        from ripple_search.workloads import make_corpus

        X, _, _ = make_corpus(100_000, 64, 50, spread=0.30, seed=0)
        bs = InstrumentedBeamSearch(n_neighbors=32, n_hyperplanes=10, n_entry=4, seed=0)
        bs.build(X)
        ceiling = 768
    return X, bs, ceiling


def run_dataset(name):
    print(f"=== {name} ===", flush=True)
    X, bs, ceiling = load_dataset(name)
    bf = BruteForce()
    bf.build(X)
    bs.X = bf.X  # share the normalized matrix

    Q, qids = hetero_queries(bf.X, N_TRAIN + N_VAL + N_TEST, seed=31)
    truth = truth_excluding_self(bf, Q, qids)
    sl_train = slice(0, N_TRAIN)
    sl_val = slice(N_TRAIN, N_TRAIN + N_VAL)
    sl_test = slice(N_TRAIN + N_VAL, None)

    # ---- collect training checkpoints at the ceiling (TRAIN queries only)
    rows_X, rows_y = [], []
    t0 = time.time()
    with threadpool_limits(limits=1):
        for q, t in zip(Q[sl_train], truth[sl_train]):
            recs = []

            def cb(feats, snap, t=t, recs=recs):
                recs.append((feats, snap()))
                return False

            bs.search(q, k=K, ef=ceiling, check_every=CHECK_EVERY, on_checkpoint=cb)
            for feats, ids in recs:
                rows_X.append(expand(feats))
                rows_y.append(recall_at_k(ids, t, K))
    Xf = np.vstack(rows_X)
    yf = np.asarray(rows_y)
    print(f" collected {len(yf)} checkpoints from {N_TRAIN} queries "
          f"in {time.time()-t0:.0f}s", flush=True)

    # ---- fit deployment (ridge) + reference (GBM)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xf)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(Xf), yf)
    gbm = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Xf, yf)
    r2_lin = ridge.score(scaler.transform(Xf), yf)
    r2_gbm = gbm.score(Xf, yf)
    print(f" fit: ridge R2={r2_lin:.3f} gbm R2={r2_gbm:.3f} (train)", flush=True)
    w = (ridge.coef_ / scaler.scale_).astype(np.float64)
    b = float(ridge.intercept_ - np.dot(ridge.coef_, scaler.mean_ / scaler.scale_))

    def run_predictor(queries, truths, tau, min_checks=2):
        recs, dists, nchecks, lat = [], [], [], []
        with threadpool_limits(limits=1):
            for q, t in zip(queries, truths):
                state = {"n": 0}

                def cb(feats, snap, state=state):
                    state["n"] += 1
                    if state["n"] < min_checks:
                        return False
                    return float(np.dot(w, expand(feats)) + b) >= tau

                t0 = time.perf_counter()
                ids, st = bs.search(q, k=K, ef=ceiling,
                                    check_every=CHECK_EVERY, on_checkpoint=cb)
                lat.append(time.perf_counter() - t0)
                recs.append(recall_at_k(ids, t, K))
                dists.append(st["dist_computations"])
                nchecks.append(st["checkpoints"])
        return {"recall": float(np.mean(recs)),
                "mean_dists": float(np.mean(dists)),
                "p95_dists": float(np.percentile(dists, 95)),
                "mean_checkpoints": float(np.mean(nchecks)),
                "p50_ms": float(np.percentile(lat, 50) * 1e3)}

    # ---- calibrate tau per target on VAL
    ceil_recs = []
    with threadpool_limits(limits=1):
        for q, t in zip(Q[sl_val], truth[sl_val]):
            r = bs.query(q, k=K, mode="fixed", ef=ceiling)
            ceil_recs.append(recall_at_k(r, t, K))
    ceiling_recall = float(np.mean(ceil_recs))
    print(f" substrate ceiling recall (ef={ceiling}, val): {ceiling_recall:.4f}", flush=True)

    frozen = {}
    for T in TARGETS:
        if T > ceiling_recall:
            frozen[T] = None
            print(f" target {T}: UNACHIEVABLE on substrate (ceiling {ceiling_recall:.4f})", flush=True)
            continue
        lo, hi = T - 0.05, 1.10
        best = None
        for _ in range(8):  # bisection on achieved val recall
            mid = (lo + hi) / 2
            r = run_predictor(Q[sl_val], truth[sl_val], tau=mid)
            if r["recall"] >= T:
                best = (mid, r)
                hi = mid
            else:
                lo = mid
        if best is None:
            best = (hi, run_predictor(Q[sl_val], truth[sl_val], tau=hi))
        frozen[T] = best[0]
        print(f" target {T}: tau={best[0]:.4f} val recall={best[1]['recall']:.4f} "
              f"dists={best[1]['mean_dists']:.0f}", flush=True)

    # ---- baselines on TEST
    def run_fixed(ef):
        recs, dists = [], []
        with threadpool_limits(limits=1):
            for q, t in zip(Q[sl_test], truth[sl_test]):
                r = bs.query(q, k=K, mode="fixed", ef=ef)
                recs.append(recall_at_k(r, t, K))
                dists.append(bs.last_dist_count)
        return float(np.mean(recs)), float(np.mean(dists))

    def run_early(ef, p):
        recs, dists = [], []
        with threadpool_limits(limits=1):
            for q, t in zip(Q[sl_test], truth[sl_test]):
                r = bs.query(q, k=K, mode="early_exit", ef=ef, patience=p)
                recs.append(recall_at_k(r, t, K))
                dists.append(bs.last_dist_count)
        return float(np.mean(recs)), float(np.mean(dists))

    fixed_rows = []
    for ef in [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]:
        if ef > ceiling:
            continue
        rec, dist = run_fixed(ef)
        fixed_rows.append({"ef": ef, "recall": rec, "mean_dists": dist})
        print(f" fixed ef={ef}: recall={rec:.4f} dists={dist:.0f}", flush=True)
    early_rows = []
    for ef in [64, 128, 192, 256, 384, 512]:
        if ef > ceiling:
            continue
        for p in [8, 16, 32, 64, 128]:
            rec, dist = run_early(ef, p)
            early_rows.append({"ef": ef, "patience": p, "recall": rec,
                               "mean_dists": dist})

    # ---- test the frozen predictor per target and score the win condition
    results = []
    for T in TARGETS:
        if frozen[T] is None:
            results.append({"target": T, "unachievable": True,
                            "ceiling_recall_val": ceiling_recall})
            continue
        pr = run_predictor(Q[sl_test], truth[sl_test], tau=frozen[T])
        fx = [r for r in fixed_rows if r["recall"] >= min(T, pr["recall"])]
        fx_best = min(fx, key=lambda r: r["mean_dists"]) if fx else None
        ee = [r for r in early_rows if r["recall"] >= min(T, pr["recall"])]
        ee_best = min(ee, key=lambda r: r["mean_dists"]) if ee else None
        row = {"target": T, "tau": frozen[T], "predictor": pr,
               "fixed_best": fx_best, "early_exit_best": ee_best}
        row["beats_fixed"] = bool(fx_best and pr["mean_dists"] < fx_best["mean_dists"]
                                  and pr["recall"] >= T)
        row["beats_early"] = bool(ee_best and pr["mean_dists"] < ee_best["mean_dists"]
                                  and pr["recall"] >= T)
        results.append(row)
        print(f" TARGET {T}: predictor recall={pr['recall']:.4f} "
              f"dists={pr['mean_dists']:.0f} | fixed {fx_best} | early {ee_best} | "
              f"WIN fixed={row['beats_fixed']} early={row['beats_early']}", flush=True)

    return {"ceiling": ceiling, "ceiling_recall_val": ceiling_recall,
            "r2_linear_train": r2_lin, "r2_gbm_train": r2_gbm,
            "n_checkpoint_rows": len(yf), "frozen_tau": frozen,
            "fixed_sweep": fixed_rows, "early_sweep": early_rows,
            "targets": results}


names = sys.argv[1:] or ["reuters_lsa256", "synthetic_hetero", "glove100"]
out = {"description": __doc__, "k": K, "check_every": CHECK_EVERY,
       "split": [N_TRAIN, N_VAL, N_TEST], "datasets": {}}
for name in names:
    out["datasets"][name] = run_dataset(name)
    save("exp10_learned_termination", out)  # checkpoint after each dataset
