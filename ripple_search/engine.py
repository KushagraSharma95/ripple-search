"""The Emergent Ripple Engine — a faithful implementation of the hypothesis.

Implements every mechanism the hypothesis describes:

1. "Nodes placed at extreme geometric corners" — every document gets a
   hypercube-corner address via sign random projections (the standard way to
   map real vectors onto hypercube vertices). Corner addresses select the
   ripple's entry points via multi-probe Hamming lookup.
2. "Energy ripple" — a query injects kinetic energy at the entry nodes; energy
   propagates along graph edges, attenuated by a decay factor, edge strength,
   and a similarity gate. Nodes whose incoming energy exceeds a threshold
   activate. The search terminates when energy dissipates below threshold.
3. "Hebbian learning / pheromone trails" — edges on paths that led to the
   final top-k results are strengthened; shortcut edges are burned between
   entry points and results ("highly traveled paths become permanent
   shortcuts"). Strengths relax back toward baseline over time.
4. "Log-time consolidation" — nodes older than a cutoff are merged into
   centroid macro-nodes in logarithmically growing age buckets; a ripple that
   reaches a macro-node may expand its members on demand.

Instrumentation: every query records language-independent work counters
(nodes visited, true distance computations, hops) so comparisons against
C-implemented baselines are not confounded by Python overhead.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class QueryStats:
    visited: int = 0
    dist_computations: int = 0
    hops: int = 0
    entry_candidates: int = 0
    activated_ids: list = field(default_factory=list)
    activated_energy: dict = field(default_factory=dict)


def _normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


class RippleEngine:
    """Energy-ripple search over a proximity graph with hypercube-corner entry."""

    def __init__(
        self,
        n_neighbors: int = 16,
        n_hyperplanes: int = 16,
        decay: float = 0.80,
        init_energy: float = 1.0,
        theta: float = 1e-4,
        gamma: float = 2.0,
        n_entry: int = 8,
        max_visits: int | None = None,
        hebbian_eta: float = 0.10,
        hebbian_cap: float = 3.0,
        hebbian_relax: float = 0.999,
        shortcut_budget: int = 8,
        seed: int = 0,
    ):
        self.n_neighbors = n_neighbors
        self.n_hyperplanes = n_hyperplanes
        self.decay = decay
        self.init_energy = init_energy
        self.theta = theta
        self.gamma = gamma
        self.n_entry = n_entry
        self.max_visits = max_visits
        self.hebbian_eta = hebbian_eta
        self.hebbian_cap = hebbian_cap
        self.hebbian_relax = hebbian_relax
        self.shortcut_budget = shortcut_budget
        self.rng = np.random.default_rng(seed)
        self.last_stats: QueryStats | None = None
        self._queries_since_relax = 0

    # ------------------------------------------------------------------ build

    def build(self, X: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        """Build the kNN proximity graph and hypercube corner addresses."""
        self.X = _normalize(X)
        n, d = self.X.shape
        self.n = n
        self.timestamps = (
            np.asarray(timestamps, dtype=np.float64)
            if timestamps is not None
            else np.zeros(n)
        )

        # --- hypercube corner addresses (sign random projection) ---
        self.H = self.rng.standard_normal((d, self.n_hyperplanes)).astype(np.float32)
        proj = self.X @ self.H
        self.codes = self._pack_codes(proj > 0)
        self.buckets: dict[int, list[int]] = {}
        for i, c in enumerate(self.codes):
            self.buckets.setdefault(int(c), []).append(i)

        # --- kNN graph (exact, brute force cosine; build cost reported separately) ---
        k = min(self.n_neighbors, n - 1)
        self.neighbor_ids = np.empty((n, k), dtype=np.int32)
        self.neighbor_base = np.empty((n, k), dtype=np.float32)
        block = max(1, min(2048, (2**27) // max(n, 1)))
        for start in range(0, n, block):
            sims = self.X[start : start + block] @ self.X.T
            for row in range(sims.shape[0]):
                sims[row, start + row] = -np.inf
            idx = np.argpartition(-sims, k, axis=1)[:, :k]
            for row in range(sims.shape[0]):
                order = np.argsort(-sims[row, idx[row]])
                self.neighbor_ids[start + row] = idx[row][order]
                self.neighbor_base[start + row] = sims[row, idx[row][order]]
        # Edge base weights in [0, 1]; Hebbian multiplier starts neutral at 1.
        self.neighbor_base = np.clip(self.neighbor_base, 0.05, None)
        self.hebb = np.ones_like(self.neighbor_base)
        # Burned shortcuts: per-node extra adjacency (id list, strength list).
        self.shortcuts: dict[int, list[tuple[int, float]]] = {}

        # Macro-node state (filled by consolidate()).
        self.macro_centroids: np.ndarray | None = None
        self.macro_members: list[list[int]] = []
        self.active_mask = np.ones(n, dtype=bool)

    @staticmethod
    def _pack_codes(bits: np.ndarray) -> np.ndarray:
        weights = (1 << np.arange(bits.shape[1], dtype=np.int64))
        return (bits.astype(np.int64) * weights).sum(axis=1)

    # ------------------------------------------------------------ entry points

    def _entry_points(self, q: np.ndarray, stats: QueryStats) -> list[int]:
        """Multi-probe Hamming lookup over hypercube corner addresses."""
        proj = q @ self.H
        code = int(self._pack_codes((proj > 0)[None, :])[0])
        # Probe the exact corner first, then corners differing in the bits whose
        # projection magnitude is smallest (least confident bits).
        order = np.argsort(np.abs(proj))
        entries: list[int] = []
        probes = [code]
        for b in order:
            probes.append(code ^ (1 << int(b)))
            if len(probes) >= 1 + self.n_hyperplanes:
                break
        for p in probes:
            for i in self.buckets.get(p, ()):
                if self.active_mask[i]:
                    entries.append(i)
                    if len(entries) >= self.n_entry:
                        stats.entry_candidates = len(entries)
                        return entries
        # Fallback: random entries so the ripple always has somewhere to start.
        tries = 0
        while len(entries) < min(self.n_entry, self.n) and tries < 50 * self.n_entry:
            tries += 1
            i = int(self.rng.integers(len(self.active_mask)))
            if self.active_mask[i] and i not in entries:
                entries.append(i)
        stats.entry_candidates = len(entries)
        return entries

    # ------------------------------------------------------------------ query

    def query(self, q: np.ndarray, k: int = 10, learn: bool = True) -> list[int]:
        q = np.asarray(q, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        stats = QueryStats()
        entries = self._entry_points(q, stats)

        settled: dict[int, float] = {}       # node -> accumulated energy
        sims: dict[int, float] = {}          # node -> true cosine (computed once)
        parent: dict[int, int] = {}          # best-energy predecessor
        heap: list[tuple[float, int, int]] = []

        e0 = self.init_energy / max(len(entries), 1)
        for i in entries:
            heapq.heappush(heap, (-e0, i, -1))

        max_visits = self.max_visits or self.n
        while heap and stats.visited < max_visits:
            neg_e, u, par = heapq.heappop(heap)
            e = -neg_e
            if u in settled:
                settled[u] += e
                continue
            if e < self.theta:
                break
            settled[u] = e
            if par >= 0:
                parent[u] = par
            s = float(self.X[u] @ q)
            sims[u] = s
            stats.visited += 1
            stats.dist_computations += 1
            # Re-radiation gate: a node re-emits energy in proportion to how
            # well it matches the query (poor matches absorb the ripple).
            gate = max(0.0, s) ** self.gamma
            if gate <= 0.0:
                continue
            out = e * self.decay * gate
            if out < self.theta:
                continue
            nbr = self.neighbor_ids[u]
            w = self.neighbor_base[u] * self.hebb[u]
            for j in range(len(nbr)):
                v = int(nbr[j])
                if v in settled or not self.active_mask[v]:
                    continue
                ev = out * float(w[j])
                if ev >= self.theta:
                    heapq.heappush(heap, (-ev, v, u))
                    stats.hops += 1
            for v, ws in self.shortcuts.get(u, ()):
                if v in settled or not self.active_mask[v]:
                    continue
                ev = out * ws
                if ev >= self.theta:
                    heapq.heappush(heap, (-ev, v, u))
                    stats.hops += 1

        # Expand any macro-nodes that activated (count the work).
        expanded: dict[int, float] = {}
        for u, e in settled.items():
            if u >= self.n:  # macro-node id space starts at n
                for m in self.macro_members[u - self.n]:
                    s = float(self.X[m] @ q)
                    stats.dist_computations += 1
                    expanded[m] = s
        for m, s in expanded.items():
            sims.setdefault(m, s)

        # Macro-nodes are routing hubs, not documents: only original ids rank.
        top = heapq.nlargest(k, ((i, s) for i, s in sims.items() if i < self.n),
                             key=lambda kv: kv[1])
        result = [i for i, _ in top]
        stats.activated_ids = list(settled.keys())
        stats.activated_energy = settled
        self.last_stats = stats

        if learn:
            self._hebbian_update(result, parent, entries)
        return result

    # --------------------------------------------------------------- learning

    def _hebbian_update(self, result: list[int], parent: dict[int, int], entries: list[int]) -> None:
        """Strengthen edges on paths that produced results; burn shortcuts."""
        for r in result:
            node, steps = r, 0
            while node in parent and steps < 64:
                p = parent[node]
                row = self.neighbor_ids[p]
                hit = np.where(row == node)[0]
                if hit.size:
                    j = int(hit[0])
                    self.hebb[p, j] = min(self.hebb[p, j] * (1 + self.hebbian_eta), self.hebbian_cap)
                node, steps = p, steps + 1
            # Burn a shortcut from the entry region straight to the result.
            if steps >= 2 and entries:
                src = entries[0]
                lst = self.shortcuts.setdefault(src, [])
                if all(v != r for v, _ in lst):
                    lst.append((r, 1.0))
                    if len(lst) > self.shortcut_budget:
                        lst.pop(0)
        # Relax all strengths toward baseline occasionally ("decay of trails").
        self._queries_since_relax += 1
        if self._queries_since_relax >= 100:
            self.hebb += (1.0 - self.hebb) * (1.0 - self.hebbian_relax ** 100)
            self._queries_since_relax = 0

    def freeze(self) -> None:
        """Disable adaptation (for ablation runs)."""
        self.hebbian_eta = 0.0
        self.shortcut_budget = 0

    # ----------------------------------------------------------- consolidation

    def consolidate(self, now: float, base_window: float, max_group: int = 32) -> int:
        """Merge old nodes into centroid macro-nodes in log-spaced age buckets.

        Nodes with age > base_window are grouped by (log2 age bucket, corner
        code prefix) and replaced by their centroid. Returns macro-node count.
        """
        ages = now - self.timestamps
        old = np.where((ages > base_window) & self.active_mask[: self.n])[0]
        groups: dict[tuple[int, int], list[int]] = {}
        for i in old:
            log_bucket = int(math.log2(max(ages[i] / base_window, 1.0)))
            prefix = int(self.codes[i]) >> max(self.n_hyperplanes - 10, 0)
            groups.setdefault((log_bucket, prefix), []).append(int(i))

        centroids, members = [], []
        for key, ids in groups.items():
            for s in range(0, len(ids), max_group):
                chunk = ids[s : s + max_group]
                if len(chunk) < 2:
                    continue
                c = self.X[chunk].mean(axis=0)
                nrm = np.linalg.norm(c)
                if nrm == 0:
                    continue
                centroids.append(c / nrm)
                members.append(chunk)
                for i in chunk:
                    self.active_mask[i] = False

        if not centroids:
            return 0
        self.macro_centroids = np.asarray(centroids, dtype=np.float32)
        self.macro_members = members
        # Macro-nodes join the graph: append rows whose neighbors are the
        # nearest active nodes and other macro-nodes.
        m = len(members)
        full = np.vstack([self.X, self.macro_centroids])
        self.active_mask = np.concatenate([self.active_mask[: self.n], np.ones(m, dtype=bool)])
        k = self.neighbor_ids.shape[1]
        sims = self.macro_centroids @ full.T
        sims[:, ~self.active_mask] = -np.inf
        for r in range(m):
            sims[r, self.n + r] = -np.inf
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        macro_nbr = np.empty((m, k), dtype=np.int32)
        macro_w = np.empty((m, k), dtype=np.float32)
        for r in range(m):
            order = np.argsort(-sims[r, idx[r]])
            macro_nbr[r] = idx[r][order]
            macro_w[r] = np.clip(sims[r, idx[r][order]], 0.05, None)
        self.neighbor_ids = np.vstack([self.neighbor_ids, macro_nbr])
        self.neighbor_base = np.vstack([self.neighbor_base, macro_w])
        self.hebb = np.vstack([self.hebb, np.ones_like(macro_w)])
        # Re-bucket macro-nodes for corner entry.
        proj = self.macro_centroids @ self.H
        mcodes = self._pack_codes(proj > 0)
        for r, c in enumerate(mcodes):
            self.buckets.setdefault(int(c), []).append(self.n + r)
        # Extend X so sims can be computed uniformly for macro ids.
        self.X = full
        # Keep the consolidated graph navigable: active originals' kNN rows
        # mostly point at now-archived nodes, so link each active original to
        # its nearest macro hubs ("the ripple hits a stable historical hub").
        act = np.where(self.active_mask[: self.n])[0]
        if len(act):
            sims_am = self.X[act] @ self.macro_centroids.T
            top2 = np.argsort(-sims_am, axis=1)[:, :2]
            for r, i in enumerate(act):
                lst = self.shortcuts.setdefault(int(i), [])
                for c in top2[r]:
                    lst.append((self.n + int(c), float(max(sims_am[r, c], 0.05))))
        return m
