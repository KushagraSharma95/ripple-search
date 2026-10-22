"""Experiment 8b — early-exit at the high-recall end (Reuters).

exp8 showed early-exit's savings GROW with the recall target on real data
(5% at 0.883 -> 24% at 0.952). This extends the sweep into the >=0.99
regime, which published ANN benchmarks mostly don't measure (they stop at
0.95) and which the frontier research flagged as under-served.
"""

import time

import numpy as np

from _common import ROOT, save, threadpool_limits
from ripple_search import BruteForce
from ripple_search.beam_search import BeamSearch
from ripple_search.timing import recall_at_k

import nltk

nltk.data.path.insert(0, str(ROOT.parent / "nltk_data"))
from nltk.corpus import reuters  # noqa: E402

from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

K = 10
EFS = [96, 128, 192, 256, 384, 512, 768]
GRID = [(ef, p) for ef in [128, 192, 256, 384, 512, 768] for p in [16, 32, 64, 128]]

texts = [reuters.raw(f) for f in sorted(reuters.fileids())]
vec = TfidfVectorizer(stop_words="english", min_df=3, max_features=50_000,
                      sublinear_tf=True)
svd = TruncatedSVD(n_components=256, random_state=0)
X = svd.fit_transform(vec.fit_transform(texts)).astype(np.float32)
X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
rng = np.random.default_rng(5)
q_ids = rng.choice(len(texts), size=300, replace=False)
Q = X[q_ids] + 0.05 * rng.standard_normal((300, 256)).astype(np.float32)
Q = (Q / np.linalg.norm(Q, axis=1, keepdims=True)).astype(np.float32)

bf = BruteForce()
bf.build(X)
truth = [bf.query(q, k=K) for q in Q]
bs = BeamSearch(n_neighbors=32, n_hyperplanes=10, n_entry=4, seed=0)
bs.build(X)


def run(mode, params):
    rows = []
    with threadpool_limits(limits=1):
        for val in params:
            rec, dists = [], []
            for q, t in zip(Q, truth):
                if mode == "fixed":
                    r = bs.query(q, k=K, mode="fixed", ef=val)
                else:
                    r = bs.query(q, k=K, mode="early_exit", ef=val[0], patience=val[1])
                rec.append(recall_at_k(r, t, K))
                dists.append(bs.last_dist_count)
            rows.append({"param": val if isinstance(val, int) else list(val),
                         "recall@10": float(np.mean(rec)),
                         "mean_dists": float(np.mean(dists)),
                         "p95_dists": float(np.percentile(dists, 95))})
            print(f"  {mode}={val}: recall={rows[-1]['recall@10']:.4f} "
                  f"dists={rows[-1]['mean_dists']:.0f}")
    return rows


print("fixed:")
fixed = run("fixed", EFS)
print("early_exit:")
early = run("early_exit", GRID)

front = sorted(early, key=lambda r: r["mean_dists"])
pf, best = [], -1.0
for p in front:
    if p["recall@10"] > best:
        pf.append(p)
        best = p["recall@10"]
dom = []
for f in fixed:
    cands = [a for a in pf if a["recall@10"] >= f["recall@10"]]
    if not cands:
        continue
    a = min(cands, key=lambda r: r["mean_dists"])
    dom.append({"recall_target": f["recall@10"], "fixed_ef": f["param"],
                "fixed_dists": f["mean_dists"], "early": a["param"],
                "early_dists": a["mean_dists"], "early_recall": a["recall@10"],
                "work_saved_pct": 100 * (1 - a["mean_dists"] / f["mean_dists"])})
    print(f"recall>={f['recall@10']:.4f}: fixed ef={f['param']} ({f['mean_dists']:.0f}) "
          f"vs {a['param']} ({a['mean_dists']:.0f}) -> saves {dom[-1]['work_saved_pct']:.0f}%")

save("exp8b_high_recall", {"description": __doc__, "fixed": fixed,
                           "early_exit": early, "dominance": dom})
