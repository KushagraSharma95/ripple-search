"""Baselines: exact brute force, HNSW (state of the art), SQLite FTS5 (real
inverted index + BM25, the architecture Oracle Text uses), PostgreSQL
full-text (a real enterprise RDBMS engine).

Why no literal Teradata / Oracle: both are proprietary licensed engines that
cannot be installed in an ephemeral container (Teradata Express is a
registration-gated VM image; Oracle Database requires accepting OTN licenses
and multi-GB images). Their text-search layers are inverted-index engines, so
SQLite FTS5 and PostgreSQL GIN+tsvector are the faithful open proxies — see
docs/METHODOLOGY.md.
"""

from __future__ import annotations

import sqlite3

import numpy as np


class BruteForce:
    """Exact cosine top-k. This defines ground truth. Work = N per query."""

    def build(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.X = X / norms
        self.n = len(X)

    def query(self, q: np.ndarray, k: int = 10) -> list[int]:
        q = np.asarray(q, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        sims = self.X @ q
        if k >= self.n:
            return list(np.argsort(-sims))
        idx = np.argpartition(-sims, k)[:k]
        return list(idx[np.argsort(-sims[idx])])


class HNSW:
    """hnswlib wrapper — the actual state of the art in graph ANN search."""

    def __init__(self, M: int = 16, ef_construction: int = 200, ef: int = 64, seed: int = 0):
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.seed = seed

    def build(self, X: np.ndarray) -> None:
        import hnswlib

        X = np.asarray(X, dtype=np.float32)
        self.index = hnswlib.Index(space="cosine", dim=X.shape[1])
        self.index.init_index(max_elements=len(X), M=self.M,
                              ef_construction=self.ef_construction,
                              random_seed=self.seed)
        self.index.add_items(X, np.arange(len(X)), num_threads=1)
        self.index.set_ef(self.ef)
        self.index.set_num_threads(1)

    def query(self, q: np.ndarray, k: int = 10) -> list[int]:
        labels, _ = self.index.knn_query(np.asarray(q, dtype=np.float32), k=k)
        return list(labels[0])


class FTS5BM25:
    """SQLite FTS5: a real C inverted index with BM25 ranking, in memory."""

    def build(self, docs: list[str]) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE docs USING fts5(body)")
        self.db.executemany("INSERT INTO docs(rowid, body) VALUES (?, ?)",
                            list(enumerate(docs)))
        self.db.commit()

    @staticmethod
    def _sanitize(query: str) -> str:
        terms = ["\"%s\"" % t.replace('"', "") for t in query.split() if t.strip('"')]
        return " OR ".join(terms) if terms else '""'

    def query(self, text: str, k: int = 10) -> list[int]:
        rows = self.db.execute(
            "SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
            (self._sanitize(text), k),
        ).fetchall()
        return [r[0] for r in rows]


class PostgresFTS:
    """PostgreSQL tsvector + GIN index with ts_rank ordering."""

    def __init__(self, dsn: str = "dbname=ripple user=postgres host=/var/run/postgresql"):
        self.dsn = dsn

    def build(self, docs: list[str]) -> None:
        import psycopg2

        self.conn = psycopg2.connect(self.dsn)
        self.conn.autocommit = True
        cur = self.conn.cursor()
        cur.execute("DROP TABLE IF EXISTS docs")
        cur.execute("CREATE TABLE docs (id int PRIMARY KEY, body text, tsv tsvector)")
        from psycopg2.extras import execute_values

        execute_values(cur, "INSERT INTO docs (id, body) VALUES %s",
                       list(enumerate(docs)), page_size=1000)
        cur.execute("UPDATE docs SET tsv = to_tsvector('english', body)")
        cur.execute("CREATE INDEX docs_tsv ON docs USING gin(tsv)")
        cur.execute("ANALYZE docs")
        self.cur = cur

    def query(self, text: str, k: int = 10) -> list[int]:
        terms = [t for t in text.replace("'", " ").split() if t]
        q = " | ".join(terms) if terms else "x"
        self.cur.execute(
            "SELECT id FROM docs, to_tsquery('english', %s) tq "
            "WHERE tsv @@ tq ORDER BY ts_rank(tsv, tq) DESC LIMIT %s",
            (q, k),
        )
        return [r[0] for r in self.cur.fetchall()]
