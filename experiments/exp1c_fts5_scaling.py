"""Experiment 1c — measured N-scaling of a real inverted index (C1b).

The hypothesis's premise: "traditional engines slow down linearly or
quadratically as data grows." C1b's refutation must be MEASURED, not argued
from complexity theory alone — so: SQLite FTS5 (a production C inverted
index) on synthetic Zipf-vocabulary documents from 2k to 200k docs, identical
10-term queries at every size, p50/p95 per size, build time separate.
"""

import time

import numpy as np

from _common import save
from ripple_search import FTS5BM25
from ripple_search.timing import summarize, time_queries

VOCAB = 30_000
DOC_LEN = 80
N_QUERIES = 200
SIZES = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]

rng = np.random.default_rng(42)
# Zipf-ish token distribution over a fixed vocabulary (natural-language-like).
ranks = np.arange(1, VOCAB + 1, dtype=np.float64)
probs = ranks ** -1.1
probs /= probs.sum()
words = np.array([f"w{i:05d}" for i in range(VOCAB)])


def make_docs(n: int) -> list[str]:
    tok = rng.choice(VOCAB, size=(n, DOC_LEN), p=probs)
    return [" ".join(words[row]) for row in tok]


rows = []
all_docs = make_docs(SIZES[-1])  # nested sizes: smaller corpora are prefixes
for n in SIZES:
    docs = all_docs[:n]
    fts = FTS5BM25()
    t0 = time.perf_counter()
    fts.build(docs)
    build_s = time.perf_counter() - t0
    # Queries: 10 consecutive tokens from random docs IN the current corpus,
    # same query seed per size so sizes are comparable.
    qr = np.random.default_rng(7)
    q_texts = []
    for _ in range(N_QUERIES):
        d = docs[qr.integers(n)].split()
        s = qr.integers(0, DOC_LEN - 10)
        q_texts.append(" ".join(d[s : s + 10]))
    res, lat = time_queries(fts.query, q_texts, k=10)
    rows.append({"n": n, "build_s": build_s, **summarize(lat)})
    print(f"n={n}: build={build_s:.1f}s p50={rows[-1]['p50_ms']:.2f}ms "
          f"p95={rows[-1]['p95_ms']:.2f}ms")

first, last = rows[0], rows[-1]
print(f"\nN grew {SIZES[-1]//SIZES[0]}x; FTS5 p50 grew "
      f"{last['p50_ms']/first['p50_ms']:.1f}x "
      f"(linear scaling would predict {SIZES[-1]//SIZES[0]}x)")
save("exp1c_fts5_scaling", {"description": __doc__, "vocab": VOCAB,
                            "doc_len": DOC_LEN, "rows": rows})
