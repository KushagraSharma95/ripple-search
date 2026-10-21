"""Instrumented vantage-point tree: counts distance computations.

This is the cleanest possible test of the hypothesis's geometric claim:

    "the higher the number of dimensions, the sharper the diagonal angles
     become. The triangle inequality math becomes far more aggressive. A
     single corner check can instantly rule out a massive percentage of the
     database."

A VP-tree prunes subtrees using EXACTLY the triangle inequality. If the claim
is right, the fraction of the database touched per query should FALL as
dimensionality rises. If the curse of dimensionality is right, it should RISE
toward 100% (brute force). We count every distance evaluation and report the
touched fraction per dimensionality.
"""

from __future__ import annotations

import numpy as np


class _Node:
    __slots__ = ("idx", "radius", "inside", "outside", "leaf")

    def __init__(self, idx=None, radius=0.0, inside=None, outside=None, leaf=None):
        self.idx = idx
        self.radius = radius
        self.inside = inside
        self.outside = outside
        self.leaf = leaf


class VPTree:
    def __init__(self, X: np.ndarray, leaf_size: int = 16, seed: int = 0):
        self.X = np.asarray(X, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.dist_count = 0
        self.root = self._build(np.arange(len(self.X)))

    def _dist(self, i: int, q: np.ndarray) -> float:
        self.dist_count += 1
        return float(np.linalg.norm(self.X[i] - q))

    def _build(self, ids: np.ndarray) -> _Node:
        if len(ids) <= 16:
            return _Node(leaf=ids)
        vp = ids[self.rng.integers(len(ids))]
        rest = ids[ids != vp]
        d = np.linalg.norm(self.X[rest] - self.X[vp], axis=1)
        median = float(np.median(d))
        inside = rest[d <= median]
        outside = rest[d > median]
        if len(inside) == 0 or len(outside) == 0:
            return _Node(leaf=ids)
        return _Node(idx=int(vp), radius=median,
                     inside=self._build(inside), outside=self._build(outside))

    def query(self, q: np.ndarray, k: int = 10) -> tuple[list[int], int]:
        """Return (top-k ids, distance computations used for this query)."""
        q = np.asarray(q, dtype=np.float32)
        start = self.dist_count
        best: list[tuple[float, int]] = []  # max-heap by negated dist

        def consider(i: int) -> None:
            d = self._dist(i, q)
            if len(best) < k:
                best.append((d, i))
                best.sort()
            elif d < best[-1][0]:
                best[-1] = (d, i)
                best.sort()

        def tau() -> float:
            return best[-1][0] if len(best) == k else np.inf

        def search(node: _Node) -> None:
            if node is None:
                return
            if node.leaf is not None:
                for i in node.leaf:
                    consider(int(i))
                return
            d = self._dist(node.idx, q)
            if len(best) < k or d < tau():
                consider_d = (d, node.idx)
                if len(best) < k:
                    best.append(consider_d)
                    best.sort()
                elif d < best[-1][0]:
                    best[-1] = consider_d
                    best.sort()
            # Triangle inequality pruning — the claim under test.
            if d < node.radius:
                search(node.inside)
                if d + tau() >= node.radius:
                    search(node.outside)
            else:
                search(node.outside)
                if d - tau() <= node.radius:
                    search(node.inside)

        search(self.root)
        return [i for _, i in best], self.dist_count - start


def relative_contrast(X: np.ndarray, queries: np.ndarray) -> float:
    """Mean (d_max - d_min) / d_min — the standard distance-concentration
    measure. Beyer et al. 1999: as D grows this ratio collapses toward 0 for
    i.i.d. data, making 'nearest' neighbors meaningless and pruning useless."""
    vals = []
    for q in queries:
        d = np.linalg.norm(X - q, axis=1)
        d = d[d > 0]
        vals.append((d.max() - d.min()) / max(d.min(), 1e-12))
    return float(np.mean(vals))


def hamming_separation(n: int, dims: int, rng) -> tuple[float, float]:
    """The hyperdimensional-computing counterpoint: for RANDOM binary codes on
    hypercube corners, relative Hamming separation IMPROVES with D
    (quasi-orthogonality). Returns (mean_pair_dist/D, std/D)."""
    codes = rng.integers(0, 2, size=(n, dims), dtype=np.int8)
    sample = rng.integers(0, n, size=(min(n, 200), 2))
    d = [(codes[i] != codes[j]).sum() / dims for i, j in sample if i != j]
    return float(np.mean(d)), float(np.std(d))
