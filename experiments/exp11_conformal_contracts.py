"""Experiment 11 — conformal declared-recall contracts (distribution-free).

exp10's controller targets MEAN recall and calibrates its stop threshold by
bisection — empirical, no guarantee, and we measured the 0.99 contract
slipping on some seeds. This experiment replaces bisection with split
conformal calibration and upgrades the contract to PER-QUERY form:

    "at least m of the true top-10 are returned, for at least a 1-delta
     fraction of queries" — distribution-free, by exchangeability.

Mechanism. During beam search the running top-k only improves, so per-query
recall at the stop point is monotone (non-decreasing) in the stop threshold
tau. For each calibration query, one instrumented ceiling run records the
predictor value p_t and the running hit count m_t at every checkpoint; the
query's critical value c_i is the smallest candidate tau (over observed p_t
values) whose induced stop point satisfies m >= m_target, or +inf if even
the ceiling run never does. tau_hat = the ceil((n+1)(1-delta))-th smallest
c_i. Choosing tau >= c_i makes query i meet the contract, so a fresh
exchangeable query meets it with probability >= 1-delta (standard split
conformal argument). If more than a delta fraction of calibration queries
are infeasible at the ceiling, tau_hat = +inf and the contract is reported
unattainable at that coverage, honestly.

Protocol (frozen before running; published win or lose):
  - fresh query seed 311, never used in any prior run or development round
  - datasets: reuters_lsa256, synthetic_hetero (GloVe's flat-graph substrate
    cannot reach contract grade and is deferred to the substrate step)
  - splits: 600 train (predictor fit, identical to exp10's features/model),
    300 calibration, 300 test; eps ~ U[0.01, 0.15]; source doc excluded
    from truth and results symmetrically
  - contracts: m_target in {8, 9, 10} of k=10, coverage 1-delta in {0.9, 0.8}
  - measured on test: empirical coverage (must be >= 1-delta up to binomial
    noise, else the method FAILS), mean recall, mean/p95 distance
    computations
  - baselines on identical test queries: (a) the cheapest fixed ef whose
    CALIBRATION coverage >= 1-delta, evaluated once on test; (b) exp10's
    mean-recall bisection controller is reported for context, not as a
    coverage competitor (it guarantees nothing per-query)

Amendments (after adversarial review, registered in commit ec1b29d before
the seed-809 run):
  - Baseline (b) was registered but never implemented. It is dropped,
    with this disclosure: it is not a coverage competitor, and exp10
    reports it on its own seeds.
  - critical_value returned +inf for a calibration trace with no
    checkpoints even when the full-budget run met the contract. The
    correct value is -inf, since the stop rule never fires and any tau
    works. The v1 behavior is conservative (it can only raise tau_hat),
    so seed-311 validity is unaffected.
  - A second fresh query seed, 809, set via the QSEED env var, runs with
    the corrected function.
  - This amendment was meant to land with the seed-809 run, but a
    whitespace mismatch made the scripted edit a silent no-op. The
    seed-809 result file therefore embeds the v1 docstring above as its
    "description"; its query_seed field (809) is authoritative.
"""

import time

import numpy as np

from _common import ROOT, save, threadpool_limits
from ripple_search import BruteForce
from ripple_search.beam_search import InstrumentedBeamSearch
from ripple_search.timing import recall_at_k

K = 10
M_TARGETS = [8, 9, 10]
DELTAS = [0.1, 0.2]
CHECK_EVERY = 8
N_TRAIN, N_CAL, N_TEST = 600, 300, 300
import os
QSEED = int(os.environ.get("QSEED", "311"))


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
    if name == "reuters_lsa256":
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
    return bs, 768


def run_dataset(name):
    print(f"=== {name} ===", flush=True)
    bs, ceiling = load_dataset(name)
    bf = BruteForce()
    bf.build(bs.X)
    bs.X = bf.X

    Q, qids = hetero_queries(bf.X, N_TRAIN + N_CAL + N_TEST, seed=QSEED)
    truth = truth_excluding_self(bf, Q, qids)
    sl_tr = slice(0, N_TRAIN)
    sl_cal = slice(N_TRAIN, N_TRAIN + N_CAL)
    sl_te = slice(N_TRAIN + N_CAL, None)

    # ---- train the predictor (same features/model as exp10's deployment)
    rows_X, rows_y = [], []
    with threadpool_limits(limits=1):
        for q, t, sid in zip(Q[sl_tr], truth[sl_tr], qids[sl_tr]):
            recs = []

            def cb(feats, snap, recs=recs):
                recs.append((feats, snap()))
                return False

            bs.search(q, k=K + 1, ef=ceiling, check_every=CHECK_EVERY, on_checkpoint=cb)
            for feats, ids in recs:
                rows_X.append(expand(feats))
                rows_y.append(recall_at_k([i for i in ids if i != sid][:K], t, K))
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Xf, yf = np.vstack(rows_X), np.asarray(rows_y)
    scaler = StandardScaler().fit(Xf)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(Xf), yf)
    w = (ridge.coef_ / scaler.scale_).astype(np.float64)
    b = float(ridge.intercept_ - np.dot(ridge.coef_, scaler.mean_ / scaler.scale_))
    print(f" predictor trained on {len(yf)} checkpoints", flush=True)

    # ---- calibration traces: (p_t, hits_t) per checkpoint per cal query
    traces = []
    with threadpool_limits(limits=1):
        for q, t, sid in zip(Q[sl_cal], truth[sl_cal], qids[sl_cal]):
            tr = []
            tset = set(t)

            def cb(feats, snap, tr=tr, tset=tset, sid=sid):
                ids = [i for i in snap() if i != sid][:K]
                hits = len(set(ids) & tset)
                tr.append((float(np.dot(w, expand(feats)) + b), hits))
                return False

            ids, st = bs.search(q, k=K + 1, ef=ceiling, check_every=CHECK_EVERY,
                                on_checkpoint=cb)
            final_ids = [i for i in ids if i != sid][:K]
            tr.append((np.inf, len(set(final_ids) & tset)))  # the no-early-stop point
            traces.append(tr)

    def critical_value(tr, m_target):
        """Smallest tau among observed p_t (or +inf) such that stopping at the
        first checkpoint with p_t >= tau yields hits >= m_target."""
        cands = sorted({p for p, _ in tr if np.isfinite(p)})
        for tau in cands:  # ascending: first feasible tau is the smallest
            for p, hits in tr:  # first checkpoint with p >= tau is the stop
                if p >= tau:
                    if hits >= m_target:
                        return tau
                    break
        if tr[-1][1] >= m_target:
            if not cands:
                return -np.inf  # no checkpoints: stop rule never fires
            # Only the no-early-stop ceiling point meets the contract: the
            # critical value is anything strictly above every observed p_t,
            # so any tau >= c_i provably runs to the bound.
            return float(np.nextafter(cands[-1], np.inf))
        return np.inf

    # ---- fixed-ef calibration coverage (for the coverage-matched baseline)
    FIXED_GRID = [16, 20, 24, 29, 35, 42, 50, 60, 72, 86, 104, 124, 149,
                  179, 215, 258, 310, 372, 446, 535, 642, 768]

    def fixed_hits(sl, ef):
        hits, dists = [], []
        with threadpool_limits(limits=1):
            for q, t, sid in zip(Q[sl], truth[sl], qids[sl]):
                r = bs.query(q, k=K + 1, mode="fixed", ef=ef)
                ids = [i for i in r if i != sid][:K]
                hits.append(len(set(ids) & set(t)))
                dists.append(bs.last_dist_count)
        return np.asarray(hits), float(np.mean(dists))

    fixed_cal = {ef: fixed_hits(sl_cal, ef) for ef in FIXED_GRID}

    results = []
    for m_target in M_TARGETS:
        cvals = np.asarray([critical_value(tr, m_target) for tr in traces])
        infeasible = float(np.mean(~np.isfinite(cvals)))
        for delta in DELTAS:
            n = len(cvals)
            rank = int(np.ceil((n + 1) * (1 - delta)))
            row = {"m_target": m_target, "delta": delta,
                   "cal_infeasible_frac": infeasible}
            if rank > n or not np.isfinite(np.sort(cvals)[rank - 1]):
                row["unattainable"] = True
                results.append(row)
                print(f" m>={m_target} 1-delta={1-delta}: UNATTAINABLE "
                      f"(infeasible frac {infeasible:.3f})", flush=True)
                continue
            tau_hat = float(np.sort(cvals)[rank - 1])

            # test with the conformal threshold
            hits_te, dists_te, lat = [], [], []
            with threadpool_limits(limits=1):
                for q, t, sid in zip(Q[sl_te], truth[sl_te], qids[sl_te]):
                    def cb(feats, snap):
                        return float(np.dot(w, expand(feats)) + b) >= tau_hat

                    t0 = time.perf_counter()
                    ids, st = bs.search(q, k=K + 1, ef=ceiling,
                                        check_every=CHECK_EVERY, on_checkpoint=cb)
                    lat.append(time.perf_counter() - t0)
                    fids = [i for i in ids if i != sid][:K]
                    hits_te.append(len(set(fids) & set(t)))
                    dists_te.append(st["dist_computations"])
            hits_te = np.asarray(hits_te)
            cov = float(np.mean(hits_te >= m_target))

            # coverage-matched fixed baseline: cheapest ef with cal coverage
            ok = [ef for ef in FIXED_GRID
                  if float(np.mean(fixed_cal[ef][0] >= m_target)) >= 1 - delta]
            fx = None
            if ok:
                ef0 = min(ok, key=lambda e: fixed_cal[e][1])
                h, d = fixed_hits(sl_te, ef0)
                fx = {"ef": ef0, "coverage": float(np.mean(h >= m_target)),
                      "mean_dists": d}
            row.update({"tau_hat": tau_hat,
                        "test_coverage": cov, "coverage_target": 1 - delta,
                        "coverage_met": bool(cov >= 1 - delta - 1.96 * np.sqrt(delta * (1 - delta) / len(hits_te))),
                        "coverage_met_point": bool(cov >= 1 - delta),
                        "mean_recall": float(np.mean(hits_te) / K),
                        "mean_dists": float(np.mean(dists_te)),
                        "p95_dists": float(np.percentile(dists_te, 95)),
                        "fixed_coverage_matched": fx,
                        "beats_fixed": bool(fx and cov >= 1 - delta
                                            and np.mean(dists_te) < fx["mean_dists"])})
            results.append(row)
            print(f" m>={m_target} 1-delta={1-delta}: tau={tau_hat:.3f} "
                  f"coverage={cov:.3f} dists={np.mean(dists_te):.0f} | "
                  f"fixed {fx} | WIN={row['beats_fixed']}", flush=True)
    return results


out = {"description": __doc__, "k": K, "query_seed": QSEED,
       "split": [N_TRAIN, N_CAL, N_TEST], "datasets": {}}
for name in ["reuters_lsa256", "synthetic_hetero"]:
    out["datasets"][name] = run_dataset(name)
    save("exp11_conformal_contracts" if QSEED == 311 else f"exp11_conformal_contracts_seed{QSEED}", out)
