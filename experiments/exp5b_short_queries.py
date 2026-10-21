"""Experiment 5b — text engines at their intended query shape.

exp5's 40-term OR queries are pathological for postings-list engines (every
term's postings list must be scanned and merged). This measures SQLite FTS5
and PostgreSQL FTS on the same 300 queries truncated to their lead 10 words —
the shape real keyword search takes — so the latency comparison does not
quietly slander the inverted-index engines.
"""

import time

import numpy as np

from _common import ROOT, save
from ripple_search import FTS5BM25, PostgresFTS
from ripple_search.timing import summarize, time_queries

import nltk

nltk.data.path.insert(0, str(ROOT.parent / "nltk_data"))
from nltk.corpus import reuters  # noqa: E402

K = 10
N_QUERIES = 300

fileids = sorted(reuters.fileids())
texts = [reuters.raw(f) for f in fileids]
cats = [set(reuters.categories(f)) for f in fileids]
rng = np.random.default_rng(5)  # identical to exp5/exp7
q_ids = rng.choice(len(texts), size=N_QUERIES, replace=False)
q_short = [" ".join(texts[i].split()[:10]) for i in q_ids]


def category_p_at_10(results):
    vals = []
    for qi, res in zip(q_ids, results):
        qc = cats[qi]
        hits = [r for r in res if r != qi][:K]
        if not hits or not qc:
            continue
        vals.append(np.mean([1.0 if (cats[r] & qc) else 0.0 for r in hits]))
    return float(np.mean(vals))


out = {"description": __doc__, "n_queries": N_QUERIES, "engines": {}}

fts = FTS5BM25()
t0 = time.perf_counter()
fts.build(texts)
b = time.perf_counter() - t0
res, lat = time_queries(fts.query, q_short, k=K + 1)
out["engines"]["sqlite_fts5_bm25"] = {"build_s": b, **summarize(lat),
                                      "category_p@10": category_p_at_10(res)}

pg = PostgresFTS()
t0 = time.perf_counter()
pg.build(texts)
b = time.perf_counter() - t0
res, lat = time_queries(pg.query, q_short, k=K + 1)
out["engines"]["postgres_fts_gin"] = {"build_s": b, **summarize(lat),
                                      "category_p@10": category_p_at_10(res)}

for name, s in out["engines"].items():
    print(f"{name} (10-word queries): p50={s['p50_ms']:.3f}ms p95={s['p95_ms']:.3f}ms "
          f"catP@10={s['category_p@10']:.3f}")
save("exp5b_short_queries", out)
