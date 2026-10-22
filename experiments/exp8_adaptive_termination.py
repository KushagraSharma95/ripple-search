"""Experiment 8 — the win attempt: adaptive early termination vs fixed ef.

Shipped graph-search libraries use one fixed search budget (ef) for every
query. Query difficulty varies wildly, so a per-query stopping rule should
dominate: easy queries stop early, hard ones keep going. Same graph, same
entry points, same code path — ONLY the termination policy differs.

Win condition: the patience policy's recall-vs-work Pareto front dominates
the fixed-ef front (fewer mean distance computations at equal or better
recall), on both synthetic (N=100k, D=64) and real text (Reuters LSA-256)
corpora. Work is counted in true similarity evaluations, so the result is
implementation-independent.
"""

import time

import numpy as np

from _common import ROOT, save, threadpool_limits
from ripple_search import BruteForce
from ripple_search.beam_search import BeamSearch
from ripple_search.timing import recall_at_k
from ripple_search.workloads import make_corpus, make_queries

K = 10
EFS = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256]
PATIENCES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def pareto(points):
    """points: list of dicts with mean_dists & recall. Returns the frontier
    (increasing work, increasing recall, no dominated points)."""
    pts = sorted(points, key=lambda r: r["mean_dists"])
    front, best = [], -1.0
    for p in pts:
        if p["recall@10"] > best:
            front.append(p)
            best = p["recall@10"]
    return front


def evaluate(bs, Q, truth, mode, params):
    rows = []
    with threadpool_limits(limits=1):
        for val in params:
            rec, dists, lat = [], [], []
            for q, t in zip(Q, truth):
                t0 = time.perf_counter()
                if mode == "fixed":
                    r = bs.query(q, k=K, mode="fixed", ef=val)
                else:
                    r = bs.query(q, k=K, mode="patience", patience=val)
                lat.append(time.perf_counter() - t0)
                rec.append(recall_at_k(r, t, K))
                dists.append(bs.last_dist_count)
            rows.append({
                "param": val,
                "recall@10": float(np.mean(rec)),
                "mean_dists": float(np.mean(dists)),
                "p95_dists": float(np.percentile(dists, 95)),
                "p50_ms": float(np.percentile(lat, 50) * 1e3),
            })
            print(f"  {mode}={val}: recall={rows[-1]['recall@10']:.3f} "
                  f"dists={rows[-1]['mean_dists']:.0f} p95={rows[-1]['p95_dists']:.0f}")
    return rows


def dominance(fixed_rows, adaptive_rows):
    """At each fixed-ef recall level, how much work does the adaptive policy
    need? Interpolate adaptive work at the fixed recall via its frontier."""
    front = pareto(adaptive_rows)
    out = []
    for f in fixed_rows:
        cands = [a for a in front if a["recall@10"] >= f["recall@10"]]
        if not cands:
            continue
        a = min(cands, key=lambda r: r["mean_dists"])
        out.append({
            "recall_target": f["recall@10"],
            "fixed_ef": f["param"], "fixed_dists": f["mean_dists"],
            "adaptive_patience": a["param"], "adaptive_dists": a["mean_dists"],
            "adaptive_recall": a["recall@10"],
            "work_saved_pct": 100 * (1 - a["mean_dists"] / f["mean_dists"]),
        })
    return out


datasets = {}

X, _, centers = make_corpus(100_000, 64, 50, spread=0.30, seed=0)
Q, _ = make_queries(centers, 300, "uniform", spread=0.30, seed=21)
datasets["synthetic_100k_d64"] = (X, Q)

try:
    import nltk

    nltk.data.path.insert(0, str(ROOT.parent / "nltk_data"))
    from nltk.corpus import reuters
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [reuters.raw(f) for f in sorted(reuters.fileids())]
    vec = TfidfVectorizer(stop_words="english", min_df=3, max_features=50_000,
                          sublinear_tf=True)
    svd = TruncatedSVD(n_components=256, random_state=0)
    Xr = svd.fit_transform(vec.fit_transform(texts)).astype(np.float32)
    Xr /= np.clip(np.linalg.norm(Xr, axis=1, keepdims=True), 1e-9, None)
    rng = np.random.default_rng(5)
    q_ids = rng.choice(len(texts), size=300, replace=False)
    Qr = Xr[q_ids] + 0.05 * rng.standard_normal((300, 256)).astype(np.float32)
    Qr /= np.linalg.norm(Qr, axis=1, keepdims=True)
    datasets["reuters_lsa256"] = (Xr, Qr.astype(np.float32))
except Exception as e:  # pragma: no cover
    print("reuters unavailable:", e)

out = {"description": __doc__, "k": K, "datasets": {}}
for name, (Xd, Qd) in datasets.items():
    print(f"=== {name} (n={len(Xd)}) ===")
    bf = BruteForce()
    bf.build(Xd)
    truth = [bf.query(q, k=K) for q in Qd]
    bs = BeamSearch(n_neighbors=32, n_hyperplanes=10, n_entry=4, seed=0)
    t0 = time.perf_counter()
    bs.build(Xd)
    build_s = time.perf_counter() - t0
    print(" fixed-ef sweep:")
    fixed = evaluate(bs, Qd, truth, "fixed", EFS)
    print(" patience sweep:")
    adaptive = evaluate(bs, Qd, truth, "patience", PATIENCES)
    dom = dominance(fixed, adaptive)
    for d in dom:
        print(f"  recall>={d['recall_target']:.3f}: fixed ef={d['fixed_ef']} "
              f"({d['fixed_dists']:.0f} dists) vs patience={d['adaptive_patience']} "
              f"({d['adaptive_dists']:.0f} dists) -> saves {d['work_saved_pct']:.0f}%")
    out["datasets"][name] = {"build_s": build_s, "fixed": fixed,
                             "adaptive": adaptive, "dominance": dom}

save("exp8_adaptive_termination", out)
