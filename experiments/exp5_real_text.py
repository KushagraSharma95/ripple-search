"""Experiment 5 — real text, real database engines. The user's question:
"is it faster than search in Teradata or Oracle databases?"

Teradata and Oracle cannot be installed here (proprietary, licensed); their
text search layers are inverted-index engines (Oracle Text CONTEXT = inverted
index + relevance ranking), so we benchmark against the same architecture in
open form, PLUS the modern ANN state of the art:

    FTS5 BM25       SQLite's C inverted index (in-memory) — proxy for
                    Oracle Text-style architecture
    PostgreSQL FTS  tsvector + GIN index + ts_rank — a real enterprise RDBMS
    brute force     exact cosine over LSA embeddings (ground truth)
    HNSW            graph ANN state of the art
    Ripple          the hypothesis

Corpus: Reuters-21578 (10,788 docs, 90 topic labels). Queries: 300 held-out
documents' lead text. Quality metrics: same-category precision@10 (label
agreement, excluding the query document itself) for ALL engines, plus
recall@10 vs exact cosine for the embedding engines. Latency: single-threaded,
p50/p95, index build reported separately.
"""

import sys
import time
from pathlib import Path

import numpy as np

from _common import ROOT, save, threadpool_limits
from ripple_search import FTS5BM25, HNSW, BruteForce, PostgresFTS, RippleEngine
from ripple_search.timing import recall_at_k, summarize, time_queries

import nltk

nltk.data.path.insert(0, str(ROOT.parent / "nltk_data"))
from nltk.corpus import reuters  # noqa: E402

K = 10
N_QUERIES = 300
SVD_DIM = 256

fileids = sorted(reuters.fileids())
texts = [reuters.raw(f) for f in fileids]
cats = [set(reuters.categories(f)) for f in fileids]
print(f"corpus: {len(texts)} docs")

# --- embeddings: TF-IDF -> LSA (TruncatedSVD) -> unit norm ---
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

t0 = time.perf_counter()
vec = TfidfVectorizer(stop_words="english", min_df=3, max_features=50_000,
                      sublinear_tf=True)
tfidf = vec.fit_transform(texts)
svd = TruncatedSVD(n_components=SVD_DIM, random_state=0)
X = svd.fit_transform(tfidf).astype(np.float32)
X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
embed_time = time.perf_counter() - t0
print(f"embeddings: {X.shape} in {embed_time:.1f}s")

# --- queries: lead text of 300 sampled docs; the doc itself is excluded from
#     quality scoring so engines can't win by returning the query verbatim ---
rng = np.random.default_rng(5)
q_ids = rng.choice(len(texts), size=N_QUERIES, replace=False)


def lead_words(t: str, n: int = 40) -> str:
    return " ".join(t.split()[:n])


q_texts = [lead_words(texts[i]) for i in q_ids]
q_tfidf = vec.transform(q_texts)
Qe = svd.transform(q_tfidf).astype(np.float32)
Qe /= np.clip(np.linalg.norm(Qe, axis=1, keepdims=True), 1e-9, None)

# Per-query embedding cost, measured one query at a time (what a live system
# pays per request — batch-amortized numbers understate it ~65x). This cost
# applies to EVERY vector engine (brute force, HNSW, ripple) on top of its
# search latency; text engines parse inside their own timed query call.
from _common import threadpool_limits as _tpl

with _tpl(limits=1):
    _lat = []
    for t in q_texts:
        t0 = time.perf_counter()
        v = svd.transform(vec.transform([t]))
        _lat.append(time.perf_counter() - t0)
q_embed_time = float(np.median(_lat))


def category_p_at_10(results: list[list[int]]) -> float:
    vals = []
    for qi, res in zip(q_ids, results):
        qc = cats[qi]
        hits = [r for r in res if r != qi][:K]
        if not hits or not qc:
            continue
        vals.append(np.mean([1.0 if (cats[r] & qc) else 0.0 for r in hits]))
    return float(np.mean(vals))


out = {"description": __doc__, "n_docs": len(texts), "n_queries": N_QUERIES,
       "svd_dim": SVD_DIM, "embed_pipeline_build_s": embed_time,
       "query_embed_ms": q_embed_time * 1e3, "engines": {}}

# ground truth + brute force
bf = BruteForce()
t0 = time.perf_counter()
bf.build(X)
b = time.perf_counter() - t0
with threadpool_limits(limits=1):
    truth, lat = time_queries(bf.query, Qe, k=K + 1)
out["engines"]["brute_force"] = {"build_s": b, **summarize(lat),
                                 "recall@10_vs_exact": 1.0,
                                 "category_p@10": category_p_at_10(truth)}

rip = RippleEngine(n_neighbors=32, n_hyperplanes=10, decay=0.90, gamma=1.0,
                   theta=1e-4, n_entry=16, init_energy=4.0, seed=0)
t0 = time.perf_counter()
rip.build(X)
b = time.perf_counter() - t0
visited = []


def rip_query(q, k=K + 1):
    r = rip.query(q, k=k, learn=False)
    visited.append(rip.last_stats.visited)
    return r


with threadpool_limits(limits=1):
    res, lat = time_queries(rip_query, Qe)
visited = visited[-len(Qe):]  # drop warmup-call counters
out["engines"]["ripple"] = {
    "build_s": b, **summarize(lat),
    "recall@10_vs_exact": float(np.mean([recall_at_k(r, t, K) for r, t in zip(res, truth)])),
    "category_p@10": category_p_at_10(res),
    "mean_visited": float(np.mean(visited)),
}

hnsw = HNSW(M=16, ef_construction=200, ef=64, seed=0)
t0 = time.perf_counter()
hnsw.build(X)
b = time.perf_counter() - t0
with threadpool_limits(limits=1):
    res, lat = time_queries(hnsw.query, Qe, k=K + 1)
out["engines"]["hnsw"] = {
    "build_s": b, **summarize(lat),
    "recall@10_vs_exact": float(np.mean([recall_at_k(r, t, K) for r, t in zip(res, truth)])),
    "category_p@10": category_p_at_10(res),
}

# --- text engines (query side uses raw text; latency includes their own parse) ---
fts = FTS5BM25()
t0 = time.perf_counter()
fts.build(texts)
b = time.perf_counter() - t0
res, lat = time_queries(fts.query, q_texts, k=K + 1)
out["engines"]["sqlite_fts5_bm25"] = {"build_s": b, **summarize(lat),
                                      "category_p@10": category_p_at_10(res)}

try:
    # Sweep ts_rank's length-normalization the way Oracle's operator was
    # swept; publish the best-quality setting and the sweep itself.
    pg = PostgresFTS()
    t0 = time.perf_counter()
    pg.build(texts)
    b = time.perf_counter() - t0
    pg.cur.execute("SET max_parallel_workers_per_gather = 0")
    sweep = {}
    best = None
    for norm in [0, 2, 32]:
        pg.normalization = norm
        res, lat = time_queries(pg.query, q_texts, k=K + 1)
        entry = {"normalization": norm, **summarize(lat),
                 "category_p@10": category_p_at_10(res)}
        sweep[norm] = entry
        if best is None or entry["category_p@10"] > best["category_p@10"]:
            best = entry
    out["engines"]["postgres_fts_gin"] = {"build_s": b, **best,
                                          "normalization_sweep": sweep}
except Exception as e:  # pragma: no cover
    out["engines"]["postgres_fts_gin"] = {"error": str(e)}
    print("postgres failed:", e, file=sys.stderr)

for name, stats in out["engines"].items():
    if "p50_ms" in stats:
        print(f"{name}: p50={stats['p50_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms "
              f"catP@10={stats.get('category_p@10', float('nan')):.3f} "
              f"recall={stats.get('recall@10_vs_exact', float('nan')):.3f}")

save("exp5_real_text", out)
