"""Greedy beam search over the kNN graph, with pluggable termination.

This is the mechanism that beat the ripple engine (directed navigation
instead of isotropic diffusion), implemented on the SAME graph and entry
points so termination policies can be compared apples-to-apples:

  - fixed:    classic HNSW-style search with a fixed beam width ef. One
              budget for every query — what shipped libraries do.
  - patience: adaptive stopping — keep expanding while the top-k improves;
              stop after `p` consecutive expansions that fail to improve the
              k-th best result. Easy queries stop early, hard queries keep
              going. The hypothesis under test: the patience Pareto front
              (recall vs distance computations) dominates the fixed-ef front.

Work accounting: every true similarity evaluation is counted once; counters
are language-independent so conclusions do not ride on Python overhead.
"""

from __future__ import annotations

import heapq

import numpy as np

from .engine import _normalize


class BeamSearch:
    def __init__(self, n_neighbors: int = 32, n_hyperplanes: int = 10,
                 n_entry: int = 4, seed: int = 0):
        self.n_neighbors = n_neighbors
        self.n_hyperplanes = n_hyperplanes
        self.n_entry = n_entry
        self.rng = np.random.default_rng(seed)
        self.last_dist_count = 0

    def build(self, X: np.ndarray) -> None:
        # Reuse the ripple engine's exact-kNN graph + corner-hash entry so the
        # comparison isolates the termination policy, nothing else.
        from .engine import RippleEngine

        e = RippleEngine(n_neighbors=self.n_neighbors,
                         n_hyperplanes=self.n_hyperplanes, seed=0)
        e.build(X)
        self.X = e.X
        self.neighbor_ids = e.neighbor_ids
        self.H = e.H
        self.buckets = e.buckets
        self.codes = e.codes
        self.n = e.n

    def _entries(self, q: np.ndarray) -> list[int]:
        proj = q @ self.H
        code = int(((proj > 0).astype(np.int64) *
                    (1 << np.arange(self.n_hyperplanes, dtype=np.int64))).sum())
        order = np.argsort(np.abs(proj))
        entries: list[int] = []
        probes = [code] + [code ^ (1 << int(b)) for b in order]
        for p in probes:
            for i in self.buckets.get(p, ()):
                entries.append(i)
                if len(entries) >= self.n_entry:
                    return entries
        while len(entries) < self.n_entry:
            entries.append(int(self.rng.integers(self.n)))
        return entries

    def query(self, q: np.ndarray, k: int = 10, mode: str = "fixed",
              ef: int = 64, patience: int = 16,
              max_expansions: int | None = None) -> list[int]:
        """mode='fixed': classic beam of width ef (ef >= k).
        mode='patience': unbounded beam; stop after `patience` consecutive
        node expansions that do not improve the k-th best similarity."""
        q = np.asarray(q, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        dist_count = 0

        visited: set[int] = set()
        cand: list[tuple[float, int]] = []    # min-heap on -sim (best first)
        res: list[tuple[float, int]] = []     # min-heap on sim (worst first)

        def eval_node(i: int) -> float:
            nonlocal dist_count
            dist_count += 1
            return float(self.X[i] @ q)

        beam = ef if mode == "fixed" else max(k, 10 ** 9)  # patience: no width cap
        for i in self._entries(q):
            if i in visited:
                continue
            visited.add(i)
            s = eval_node(i)
            heapq.heappush(cand, (-s, i))
            heapq.heappush(res, (s, i))
            if len(res) > beam:
                heapq.heappop(res)

        # Track the k-th best efficiently with a fixed-size heap.
        topk: list[float] = sorted(s for s, _ in res)[-k:]
        heapq.heapify(topk)

        stall = 0
        expansions = 0
        limit = max_expansions or self.n
        while cand and expansions < limit:
            neg_s, u = heapq.heappop(cand)
            s_u = -neg_s
            if mode == "fixed" and len(res) >= beam and s_u < res[0][0]:
                break  # classic bound: best candidate worse than beam's worst
            if mode == "patience" and len(topk) >= k and s_u < topk[0] and stall >= patience:
                break
            expansions += 1
            improved = False
            for v in self.neighbor_ids[u]:
                v = int(v)
                if v in visited:
                    continue
                visited.add(v)
                s = eval_node(v)
                if mode == "fixed":
                    if len(res) < beam or s > res[0][0]:
                        heapq.heappush(cand, (-s, v))
                        heapq.heappush(res, (s, v))
                        if len(res) > beam:
                            heapq.heappop(res)
                else:
                    heapq.heappush(cand, (-s, v))
                    heapq.heappush(res, (s, v))
                if len(topk) < k:
                    heapq.heappush(topk, s)
                    improved = True
                elif s > topk[0]:
                    heapq.heapreplace(topk, s)
                    improved = True
            if mode == "patience":
                stall = 0 if improved else stall + 1
                if stall >= patience and len(topk) >= k:
                    break

        self.last_dist_count = dist_count
        return [i for _, i in heapq.nlargest(k, res)]
