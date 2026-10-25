"""Experiment 13 — does the learned controller transfer across corpora?

exp10/11/12 train a recall predictor per corpus. If search-dynamics
features generalize, the expensive part (labeled checkpoint collection)
need not be repeated per deployment — only the cheap tau calibration on
local validation queries. Design:

  - corpora: synthetic (100k, d=64) and Reuters LSA (10.8k, d=256) — very
    different scale and dimensionality, same feature extractor
  - directions: synthetic->reuters and reuters->synthetic
  - the transferred predictor is trained on the SOURCE train split; tau is
    always calibrated on the TARGET val split (realistic deployment);
    evaluation once on the TARGET test split
  - feature ablation: full 15 features vs the scale-free subset (ratios,
    similarities, budget fractions — no raw node counts), registered
    prediction: scale-free transfers better
  - reference: the natively trained controller (target train split)
  - shared baseline: cheapest val-selected fixed ef; the reported quantity
    per (direction, target, feature set) is the dividend vs that baseline
    and the transfer cost = native dividend - transferred dividend

Fresh query seed 523; targets 0.90 and 0.95; published win or lose.

v2 amendment (after adversarial review, registered before the v2 runs):
the v1 tau bisection was bounded to [T-0.05, 1.10]; every synthetic->reuters
cell — native included — was pinned at that floor, so local calibration
could never trade the transferred models' recall overshoot for work, and
the v1 transfer magnitudes were partly a protocol artifact. v2 widens the
search to [-0.5, 1.5] with 14 bisection steps, records a floor-hit flag per
cell, and adds `transfer_restd`: the source-trained ridge with feature
standardization re-fitted on UNLABELED target train-split features (no
target ground truth used — deployable). Fresh seed 617 via QSEED.
Registered predictions: (a) widened calibration shrinks the
synthetic->reuters failure; (b) re-standardization recovers part of the
gap to native.
"""

import numpy as np

from _common import ROOT, save, threadpool_limits
from ripple_search import BruteForce
from ripple_search.beam_search import InstrumentedBeamSearch
from ripple_search.timing import recall_at_k

K = 10
TARGETS = [0.90, 0.95]
CHECK_EVERY = 8
N_TRAIN, N_VAL, N_TEST = 600, 300, 300
import os
QSEED = int(os.environ.get("QSEED", "523"))
CEILING = 768
SCALE_FREE = [3, 4, 5, 6, 7, 8, 11, 12, 13, 14]


def expand(feats):
    f = np.asarray(feats, dtype=np.float64)
    e, v, stall, imp, top, kth, ratio, gap, frac = f
    return np.array([
        e, v, stall, imp, top, kth, ratio, gap, frac,
        np.log1p(e), np.log1p(stall), stall / max(e, 1.0),
        float(gap < 0), gap * frac, ratio * ratio,
    ])


def load(name):
    if name == "reuters":
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
    else:
        from ripple_search.workloads import make_corpus

        X, _, _ = make_corpus(100_000, 64, 50, spread=0.30, seed=0)
    bs = InstrumentedBeamSearch(n_neighbors=32, n_hyperplanes=10, n_entry=4, seed=0)
    bs.build(X)
    bf = BruteForce()
    bf.build(X)
    bs.X = bf.X
    rng = np.random.default_rng(QSEED)
    n = N_TRAIN + N_VAL + N_TEST
    ids = rng.choice(len(bf.X), size=n, replace=False)
    eps = rng.uniform(0.01, 0.15, n).astype(np.float32)
    Q = bf.X[ids] + eps[:, None] * rng.standard_normal((n, bf.X.shape[1])).astype(np.float32)
    Q = (Q / np.linalg.norm(Q, axis=1, keepdims=True)).astype(np.float32)
    truth = []
    for q, sid in zip(Q, ids):
        r = bf.query(q, k=K + 1)
        truth.append([i for i in r if i != sid][:K])
    return {"bs": bs, "bf": bf, "Q": Q, "ids": ids, "truth": truth}


def collect(ds):
    rows_X, rows_y = [], []
    bs = ds["bs"]
    with threadpool_limits(limits=1):
        for q, t, sid in zip(ds["Q"][:N_TRAIN], ds["truth"][:N_TRAIN], ds["ids"][:N_TRAIN]):
            recs = []

            def cb(feats, snap, recs=recs):
                recs.append((feats, snap()))
                return False

            bs.search(q, k=K + 1, ef=CEILING, check_every=CHECK_EVERY, on_checkpoint=cb)
            for feats, iids in recs:
                rows_X.append(expand(feats))
                rows_y.append(recall_at_k([i for i in iids if i != sid][:K], t, K))
    return np.vstack(rows_X), np.asarray(rows_y)


def fit(Xf, yf, cols=None, restd_X=None):
    """Ridge on standardized features, folded into (w, b). With restd_X,
    the learned coefficients are re-expressed against the mean/scale of
    restd_X (unlabeled target features) instead of the source's."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Xs = Xf[:, cols] if cols is not None else Xf
    scaler = StandardScaler().fit(Xs)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(Xs), yf)
    if restd_X is not None:
        Xt = restd_X[:, cols] if cols is not None else restd_X
        scaler = StandardScaler().fit(Xt)
    w = (ridge.coef_ / scaler.scale_).astype(np.float64)
    b = float(ridge.intercept_ - np.dot(ridge.coef_, scaler.mean_ / scaler.scale_))
    return w, b, cols


def run_pred(ds, sl, model, tau):
    w, b, cols = model
    bs = ds["bs"]
    recs, dists = [], []
    with threadpool_limits(limits=1):
        for q, t, sid in zip(ds["Q"][sl], ds["truth"][sl], ds["ids"][sl]):
            def cb(feats, snap):
                fx = expand(feats)
                if cols is not None:
                    fx = fx[cols]
                return float(np.dot(w, fx) + b) >= tau

            iids, st = bs.search(q, k=K + 1, ef=CEILING,
                                 check_every=CHECK_EVERY, on_checkpoint=cb)
            recs.append(recall_at_k([i for i in iids if i != sid][:K], t, K))
            dists.append(st["dist_computations"])
    return float(np.mean(recs)), float(np.mean(dists))


def run_fixed(ds, sl, ef):
    bs = ds["bs"]
    recs, dists = [], []
    with threadpool_limits(limits=1):
        for q, t, sid in zip(ds["Q"][sl], ds["truth"][sl], ds["ids"][sl]):
            r = bs.query(q, k=K + 1, mode="fixed", ef=ef)
            recs.append(recall_at_k([i for i in r if i != sid][:K], t, K))
            dists.append(bs.last_dist_count)
    return float(np.mean(recs)), float(np.mean(dists))


LO, HI, STEPS = -0.5, 1.5, 14


def calibrate(ds, model, T):
    """Smallest tau (to bisection resolution) whose VAL mean recall >= T.
    Returns (tau, floored) — floored flags a tau pinned at the search floor."""
    sl = slice(N_TRAIN, N_TRAIN + N_VAL)
    lo, hi = LO, HI
    best = None
    for _ in range(STEPS):
        mid = (lo + hi) / 2
        rec, _ = run_pred(ds, sl, model, mid)
        if rec >= T:
            best = mid
            hi = mid
        else:
            lo = mid
    tau = best if best is not None else HI
    return tau, bool(tau - LO < 2 * (HI - LO) / 2 ** STEPS)


print("loading corpora and collecting training checkpoints...", flush=True)
data = {name: load(name) for name in ["synthetic", "reuters"]}
coll = {name: collect(ds) for name, ds in data.items()}
for name in data:
    print(f" {name}: {len(coll[name][1])} checkpoint rows", flush=True)

FIXED_GRID = [16, 20, 24, 29, 35, 42, 50, 60, 72, 86, 104, 124, 149,
              179, 215, 258, 310, 372, 446, 535, 642, 768]
sl_val = slice(N_TRAIN, N_TRAIN + N_VAL)
sl_te = slice(N_TRAIN + N_VAL, None)

out = {"description": __doc__, "query_seed": QSEED, "directions": {}}
for src, tgt in [("synthetic", "reuters"), ("reuters", "synthetic")]:
    print(f"=== {src} -> {tgt} ===", flush=True)
    models = {
        "native_full": fit(*coll[tgt], cols=None),
        "transfer_full": fit(*coll[src], cols=None),
        "transfer_scalefree": fit(*coll[src], cols=SCALE_FREE),
        "transfer_restd": fit(*coll[src], cols=None, restd_X=coll[tgt][0]),
    }
    fixed_val = {ef: run_fixed(data[tgt], sl_val, ef) for ef in FIXED_GRID}
    rows = []
    for T in TARGETS:
        ok = [ef for ef in FIXED_GRID if fixed_val[ef][0] >= T]
        if not ok:
            rows.append({"target": T, "unachievable": True})
            continue
        ef0 = min(ok, key=lambda e: fixed_val[e][1])
        f_rec, f_dist = run_fixed(data[tgt], sl_te, ef0)
        cell = {"target": T, "fixed_ef": ef0, "fixed_recall": f_rec,
                "fixed_dists": f_dist, "models": {}}
        for mname, model in models.items():
            tau, floored = calibrate(data[tgt], model, T)
            rec, dist = run_pred(data[tgt], sl_te, model, tau)
            cell["models"][mname] = {
                "tau": tau, "tau_floored": floored, "recall": rec, "dists": dist,
                "contract_met": bool(rec >= T),
                "dividend": 1 - dist / f_dist}
            print(f" T={T} {mname}: recall={rec:.4f} dists={dist:.0f} "
                  f"dividend={100*(1-dist/f_dist):+.1f}%", flush=True)
        rows.append(cell)
    out["directions"][f"{src}->{tgt}"] = rows
    save("exp13_transfer" if QSEED == 523 else f"exp13_transfer_seed{QSEED}", out)
