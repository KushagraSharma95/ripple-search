"""Synthetic corpora and query workloads with controllable structure.

Synthetic data is a Gaussian mixture over topic centers on the unit sphere —
realistic cluster structure (real embeddings are clustered, not i.i.d.), with
controllable N, D, topic count, and within-topic spread.

Query workloads model the hypothesis's own scenario:
  - uniform: every topic equally likely (no locality to exploit)
  - zipf: a few topics dominate (locality: adaptation/caching should help)
  - drifting: the "train of thought" — a random walk over topics where
    consecutive queries are semantically nearby (the hypothesis's key bet)
"""

from __future__ import annotations

import numpy as np


def make_corpus(n: int, d: int, n_topics: int, spread: float = 0.25, seed: int = 0):
    """Returns (X unit-normalized, topic_of_doc, topic_centers)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_topics, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    topics = rng.integers(0, n_topics, size=n)
    X = centers[topics] + spread * rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, topics, centers


def make_queries(centers: np.ndarray, n_queries: int, workload: str,
                 spread: float = 0.25, zipf_s: float = 1.3,
                 drift_step: float = 0.15, seed: int = 1):
    """Returns (Q unit-normalized, topic_of_query)."""
    rng = np.random.default_rng(seed)
    n_topics, d = centers.shape
    if workload == "uniform":
        t = rng.integers(0, n_topics, size=n_queries)
    elif workload == "zipf":
        ranks = np.arange(1, n_topics + 1, dtype=np.float64)
        p = ranks ** (-zipf_s)
        p /= p.sum()
        perm = rng.permutation(n_topics)
        t = perm[rng.choice(n_topics, size=n_queries, p=p)]
    elif workload == "drifting":
        # Random walk over topics via nearest-center transitions: consecutive
        # queries land on semantically adjacent topics.
        sims = centers @ centers.T
        np.fill_diagonal(sims, -np.inf)
        nearest = np.argsort(-sims, axis=1)[:, :5]
        t = np.empty(n_queries, dtype=np.int64)
        t[0] = rng.integers(n_topics)
        for i in range(1, n_queries):
            if rng.random() < drift_step:
                t[i] = nearest[t[i - 1]][rng.integers(5)]
            else:
                t[i] = t[i - 1]
    else:
        raise ValueError(workload)
    Q = centers[t] + spread * rng.standard_normal((n_queries, d)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return Q.astype(np.float32), t
