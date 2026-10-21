"""Experiment 7 — the real Oracle Text engine, directly.

Oracle Database Free (gvenzl/oracle-free container image) ships under the
Free Use Terms and Conditions (FUTC, embedded in the image at
$ORACLE_HOME/LICENSE, last updated 8 Oct 2018). Unlike Oracle's OTN developer
license, the FUTC contains NO benchmark-disclosure ("DeWitt") clause — the
full text was read before publishing these numbers; a copy of the relevant
grant is quoted in docs/METHODOLOGY.md.

Same corpus, queries and metrics as exp5 (Reuters-21578, 300 lead-text
queries, category precision@10, single-threaded client): a CONTEXT inverted
index with CONTAINS + SCORE ranking. Latency includes a localhost-bridge TCP
round trip (documented; FTS5 is in-process, PostgreSQL uses a unix socket).

Env: ORACLE_DSN (host:port/service), ORACLE_USER, ORACLE_PASSWORD.
"""

import os
import re
import time

import numpy as np

from _common import ROOT, save
from ripple_search.timing import summarize, time_queries

import nltk

nltk.data.path.insert(0, str(ROOT.parent / "nltk_data"))
from nltk.corpus import reuters  # noqa: E402

import oracledb  # noqa: E402

K = 10
N_QUERIES = 300

fileids = sorted(reuters.fileids())
texts = [reuters.raw(f) for f in fileids]
cats = [set(reuters.categories(f)) for f in fileids]

rng = np.random.default_rng(5)  # identical to exp5
q_ids = rng.choice(len(texts), size=N_QUERIES, replace=False)


def lead_words(t: str, n: int = 40) -> str:
    return " ".join(t.split()[:n])


q_texts = [lead_words(texts[i]) for i in q_ids]

conn = oracledb.connect(
    user=os.environ.get("ORACLE_USER", "system"),
    password=os.environ.get("ORACLE_PASSWORD", "ripple"),
    dsn=os.environ.get("ORACLE_DSN", "127.0.0.1:1521/FREEPDB1"),
)
cur = conn.cursor()

# --- load corpus ---
for stmt in ["DROP INDEX docs_ctx", "DROP TABLE docs PURGE"]:
    try:
        cur.execute(stmt)
    except oracledb.DatabaseError:
        pass
cur.execute("CREATE TABLE docs (id NUMBER PRIMARY KEY, body CLOB)")
t0 = time.perf_counter()
rows = list(enumerate(texts))
for s in range(0, len(rows), 500):
    cur.executemany("INSERT INTO docs (id, body) VALUES (:1, :2)", rows[s : s + 500])
conn.commit()
load_s = time.perf_counter() - t0
print(f"loaded {len(rows)} docs in {load_s:.1f}s")

# --- Oracle Text CONTEXT index (the inverted index Oracle Text searches) ---
t0 = time.perf_counter()
cur.execute("CREATE INDEX docs_ctx ON docs(body) INDEXTYPE IS CTXSYS.CONTEXT")
build_s = time.perf_counter() - t0
print(f"CONTEXT index built in {build_s:.1f}s")

TOKEN = re.compile(r"[A-Za-z0-9]+")


def to_contains(text: str) -> str:
    toks = TOKEN.findall(text)
    return " | ".join("{%s}" % t for t in toks) if toks else "{x}"


def oracle_query(text: str, k: int = K + 1) -> list[int]:
    try:
        cur.execute(
            "SELECT id FROM docs WHERE CONTAINS(body, :q, 1) > 0 "
            "ORDER BY SCORE(1) DESC FETCH FIRST :k ROWS ONLY",
            q=to_contains(text), k=k,
        )
        return [r[0] for r in cur.fetchall()]
    except oracledb.DatabaseError:
        return []


def category_p_at_10(results: list[list[int]]) -> float:
    vals = []
    for qi, res in zip(q_ids, results):
        qc = cats[qi]
        hits = [r for r in res if r != qi][:K]
        if not hits or not qc:
            continue
        vals.append(np.mean([1.0 if (cats[r] & qc) else 0.0 for r in hits]))
    return float(np.mean(vals))


res, lat = time_queries(oracle_query, q_texts)
out = {
    "description": __doc__,
    "engine": "oracle_text_context",
    "oracle_version": None,
    "n_docs": len(texts),
    "n_queries": N_QUERIES,
    "load_s": load_s,
    "index_build_s": build_s,
    **summarize(lat),
    "category_p@10": category_p_at_10(res),
    "empty_result_queries": sum(1 for r in res if not r),
}
cur.execute("SELECT banner_full FROM v$version")
out["oracle_version"] = cur.fetchone()[0]
print({k: v for k, v in out.items() if k != "description"})
save("exp7_oracle_text", out)
