"""Experiment 10 — learned declared-recall termination (the DARTH-protocol attack).

A small regressor predicts the search's CURRENT recall from cheap dynamics
features at checkpoints; the search stops when the predicted recall clears
the declared target. "Meets the contract" means MEAN test recall >= target
(the same mean-recall reading DARTH's declared targets use), not a per-query
guarantee.

Protocol (v2 — frozen before the final runs):
  - queries: 1,200 per dataset, heterogeneous difficulty: per-query noise
    eps ~ U[0.01, 0.15] around unit-norm corpus vectors, source doc excluded
    from truth AND from every method's results symmetrically. At the low-eps
    end queries are near-duplicates of indexed vectors, the regime most
    favorable to early termination; this is disclosed, not hidden.
  - split 600 train / 300 val / 300 test. The QUERY SEED is taken from the
    environment (QSEED); the published runs use two fresh seeds (137, 211)
    never evaluated during development.
  - training on train only; stop thresholds tau calibrated per target by
    bisection on val; baselines (fixed ef on a x1.2 grid, heuristic
    early-exit on an (ef, patience) grid) are ALSO selected on val — the
    cheapest config whose val recall clears the target — and then evaluated
    once on test, identically to the predictor.
  - a target above the substrate's val ceiling is reported unachievable.

Development-history disclosure: three earlier development rounds ran with
query seed 31 and are preserved in git history (commits 487efa0, ce2de16,
6c0076b, 2c85e36). Between those rounds the difficulty spread was reduced
(U[0.05,0.5] intended, actually U[0.02,0.25], then U[0.01,0.15]), the GloVe
graph was rebuilt (degree 32->48, entry probes 4->8), and a GBM variant was
added — i.e., the seed-31 "test" split influenced development and its
numbers are NOT publication-grade. An earlier docstring also misstated the
eps range. The published results come only from the fresh-seed runs under
this frozen protocol, reported per variant (ridge deployment / GBM
reference), win or lose, on both seeds.
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
import os
QSEED = int(os.environ.get("QSEED", "137"))


def hetero_queries(X, n, seed):
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(X), size=n, replace=False)
    eps = rng.uniform(0.01, 0.15, size=n).astype(np.float32)
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
        nbr = np.load(GLOVE_DIR / "glove100_knn48.npy")
        bs = InstrumentedBeamSearch(n_neighbors=48, n_hyperplanes=12, n_entry=8, seed=0)
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

    Q, qids = hetero_queries(bf.X, N_TRAIN + N_VAL + N_TEST, seed=QSEED)
    truth = truth_excluding_self(bf, Q, qids)
    # Every search retrieves K+1 and drops the query's source doc, matching
    # the self-excluded ground truth; otherwise the source doc occupies one
    # result slot and recall is structurally capped at (K-1)/K = 0.9.
    sl_train = slice(0, N_TRAIN)
    sl_val = slice(N_TRAIN, N_TRAIN + N_VAL)
    sl_test = slice(N_TRAIN + N_VAL, None)

    # ---- collect training checkpoints at the ceiling (TRAIN queries only)
    rows_X, rows_y = [], []
    t0 = time.time()
    with threadpool_limits(limits=1):
        for q, t, sid in zip(Q[sl_train], truth[sl_train], qids[sl_train]):
            recs = []

            def cb(feats, snap, recs=recs):
                recs.append((feats, snap()))
                return False

            bs.search(q, k=K + 1, ef=ceiling, check_every=CHECK_EVERY, on_checkpoint=cb)
            for feats, ids in recs:
                rows_X.append(expand(feats))
                rows_y.append(recall_at_k([i for i in ids if i != sid][:K], t, K))
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

    def ridge_fn(fx):
        return float(np.dot(w, fx) + b)

    def gbm_fn(fx):
        return float(gbm.predict(fx[None])[0])

    def run_predictor(queries, truths, sids, tau, min_checks=2, predict_fn=None):
        predict_fn = predict_fn or ridge_fn
        recs, dists, nchecks, lat = [], [], [], []
        with threadpool_limits(limits=1):
            for q, t, sid in zip(queries, truths, sids):
                state = {"n": 0}

                def cb(feats, snap, state=state):
                    state["n"] += 1
                    if state["n"] < min_checks:
                        return False
                    return predict_fn(expand(feats)) >= tau

                t0 = time.perf_counter()
                ids, st = bs.search(q, k=K + 1, ef=ceiling,
                                    check_every=CHECK_EVERY, on_checkpoint=cb)
                lat.append(time.perf_counter() - t0)
                recs.append(recall_at_k([i for i in ids if i != sid][:K], t, K))
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
        for q, t, sid in zip(Q[sl_val], truth[sl_val], qids[sl_val]):
            r = bs.query(q, k=K + 1, mode="fixed", ef=ceiling)
            ceil_recs.append(recall_at_k([i for i in r if i != sid][:K], t, K))
    ceiling_recall = float(np.mean(ceil_recs))
    print(f" substrate ceiling recall (ef={ceiling}, val): {ceiling_recall:.4f}", flush=True)

    variants = {"ridge": ridge_fn, "gbm": gbm_fn}
    frozen = {v: {} for v in variants}
    for vname, fn in variants.items():
        for T in TARGETS:
            if T > ceiling_recall:
                frozen[vname][T] = None
                if vname == "ridge":
                    print(f" target {T}: UNACHIEVABLE on substrate (ceiling {ceiling_recall:.4f})", flush=True)
                continue
            lo, hi = T - 0.05, 1.10
            best = None
            for _ in range(8):  # bisection on achieved val recall
                mid = (lo + hi) / 2
                r = run_predictor(Q[sl_val], truth[sl_val], qids[sl_val], tau=mid, predict_fn=fn)
                if r["recall"] >= T:
                    best = (mid, r)
                    hi = mid
                else:
                    lo = mid
            if best is None:
                best = (hi, run_predictor(Q[sl_val], truth[sl_val], qids[sl_val], tau=hi, predict_fn=fn))
            frozen[vname][T] = best[0]
            print(f" [{vname}] target {T}: tau={best[0]:.4f} val recall={best[1]['recall']:.4f} "
                  f"dists={best[1]['mean_dists']:.0f}", flush=True)

    # ---- baselines on TEST
    def run_baseline(sl, mode, ef, p=0):
        recs, dists = [], []
        with threadpool_limits(limits=1):
            for q, t, sid in zip(Q[sl], truth[sl], qids[sl]):
                if mode == "fixed":
                    r = bs.query(q, k=K + 1, mode="fixed", ef=ef)
                else:
                    r = bs.query(q, k=K + 1, mode="early_exit", ef=ef, patience=p)
                recs.append(recall_at_k([i for i in r if i != sid][:K], t, K))
                dists.append(bs.last_dist_count)
        return float(np.mean(recs)), float(np.mean(dists))

    # x1.2 fixed grid and widened early grid, SELECTED ON VAL per target.
    FIXED_GRID = [16, 20, 24, 29, 35, 42, 50, 60, 72, 86, 104, 124, 149,
                  179, 215, 258, 310, 372, 446, 535, 642, 768]
    EARLY_GRID = [(ef, p) for ef in [64, 96, 128, 192, 256, 384, 512, 768]
                  for p in [4, 8, 16, 32, 64, 128]]
    fixed_val = []
    for ef in FIXED_GRID:
        if ef > ceiling:
            continue
        rec, dist = run_baseline(sl_val, "fixed", ef)
        fixed_val.append({"ef": ef, "recall": rec, "mean_dists": dist})
    early_val = []
    for ef, pat in EARLY_GRID:
        if ef > ceiling:
            continue
        rec, dist = run_baseline(sl_val, "early", ef, pat)
        early_val.append({"ef": ef, "patience": pat, "recall": rec,
                          "mean_dists": dist})

    # ---- test the frozen predictor per target and score the win condition
    results = []
    for T in TARGETS:
        if frozen["ridge"][T] is None:
            results.append({"target": T, "unachievable": True,
                            "ceiling_recall_val": ceiling_recall})
            continue
        # Baseline configs chosen on VAL (cheapest clearing T), like tau.
        fxv = [r for r in fixed_val if r["recall"] >= T]
        fx_cfg = min(fxv, key=lambda r: r["mean_dists"]) if fxv else None
        eev = [r for r in early_val if r["recall"] >= T]
        ee_cfg = min(eev, key=lambda r: r["mean_dists"]) if eev else None
        fx_test = None
        if fx_cfg:
            rec, dist = run_baseline(sl_test, "fixed", fx_cfg["ef"])
            fx_test = {"ef": fx_cfg["ef"], "recall": rec, "mean_dists": dist}
        ee_test = None
        if ee_cfg:
            rec, dist = run_baseline(sl_test, "early", ee_cfg["ef"], ee_cfg["patience"])
            ee_test = {"ef": ee_cfg["ef"], "patience": ee_cfg["patience"],
                       "recall": rec, "mean_dists": dist}
        row = {"target": T, "fixed_val_choice": fx_cfg, "fixed_test": fx_test,
               "early_val_choice": ee_cfg, "early_test": ee_test, "variants": {}}
        for vname, fn in variants.items():
            pr = run_predictor(Q[sl_test], truth[sl_test], qids[sl_test],
                               tau=frozen[vname][T], predict_fn=fn)
            vrow = {"tau": frozen[vname][T], "predictor": pr,
                    "meets_contract": bool(pr["recall"] >= T),
                    "beats_fixed": bool(fx_test and pr["recall"] >= T
                                        and pr["mean_dists"] < fx_test["mean_dists"]),
                    "beats_early": bool(ee_test and pr["recall"] >= T
                                        and pr["mean_dists"] < ee_test["mean_dists"])}
            row["variants"][vname] = vrow
            print(f" TARGET {T} [{vname}]: recall={pr['recall']:.4f} "
                  f"dists={pr['mean_dists']:.0f} | fixed@test {fx_test} | "
                  f"early@test {ee_test} | WIN fixed={vrow['beats_fixed']} "
                  f"early={vrow['beats_early']}", flush=True)
        results.append(row)

    return {"ceiling": ceiling, "ceiling_recall_val": ceiling_recall,
            "r2_linear_train": r2_lin, "r2_gbm_train": r2_gbm,
            "n_checkpoint_rows": len(yf), "frozen_tau": frozen,
            "fixed_val_sweep": fixed_val, "early_val_sweep": early_val,
            "targets": results}


names = sys.argv[1:] or ["reuters_lsa256", "synthetic_hetero", "glove100"]
out = {"description": __doc__, "k": K, "check_every": CHECK_EVERY,
       "split": [N_TRAIN, N_VAL, N_TEST], "query_seed": QSEED, "datasets": {}}
for name in names:
    out["datasets"][name] = run_dataset(name)
    save(f"exp10_learned_termination_seed{QSEED}", out)  # checkpoint per dataset
