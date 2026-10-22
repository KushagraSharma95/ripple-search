"""Hybrid cache: LRU window + frequency-gated main region + semantic prefetch.

exp4 found specialists: ghost-frequency admission wins zipf, bounded
semantic prefetch wins drifting, uniform ties. A first frequency-only
hybrid won zipf but lost drifting (results/exp9_hybrid_tuning.json, grid
rows without a window) — the textbook sluggishness of frequency gates under
drift. The fix is the W-TinyLFU structure (Einziger et al.): a small LRU
WINDOW admits every new item (recency-friendly), and only window evictees
must pass the frequency gate into the protected MAIN region. On top of
that, this policy adds a budget-bounded semantic prefetch pool fed by the
search engine's activation energies:

  - window (window_frac of capacity): plain LRU, unconditional admission
  - main: ghost-frequency-gated, lazy min-heap eviction, O(log n)
  - prefetch pool (inside main, <= prefetch_frac of capacity): top
    `prefetch_m` neighbors by energy; the pool evicts by RECENCY (fresh
    prefetches displace stale ones — under drift the newest neighbors are
    the likely future needs), never displaces proven residents, and a
    prefetch graduates to main on first hit

Tuning protocol: knobs chosen on seed 0 only; validated untouched on
seeds 1-2 (exp9).
"""

from __future__ import annotations

import heapq
from collections import OrderedDict, defaultdict

from .thermo_cache import BaseCache


class HybridCache(BaseCache):
    def __init__(self, capacity: int, prefetch_m: int = 100,
                 prefetch_frac: float = 0.5, window_frac: float = 0.1,
                 age_every: int = 2000):
        super().__init__(capacity)
        self.window_size = max(1, int(window_frac * capacity))
        self.main_cap = max(1, capacity - self.window_size)
        self.prefetch_m = prefetch_m
        self.prefetch_budget = max(1, int(prefetch_frac * self.main_cap))
        self.age_every = age_every
        self.freq: dict = defaultdict(int)   # ghost frequencies (all keys)
        self.window: OrderedDict = OrderedDict()
        self.main: set = set()
        self.prefetched: OrderedDict = OrderedDict()  # unproven prefetches, LRU order
        self._mheap: list = []               # lazy (freq, seq, key) over main
        self._seq = 0
        self._ops = 0

    # ----------------------------------------------------------- heap helpers
    def _push(self, k) -> None:
        self._seq += 1
        heapq.heappush(self._mheap, (self.freq[k], self._seq, k))

    def _peek(self, heap, pool):
        while heap:
            f, _, k = heap[0]
            if k not in pool or self.freq[k] != f:
                heapq.heappop(heap)
                continue
            return f, k
        return None

    # ------------------------------------------------------------- interface
    def contains(self, key, now) -> bool:
        if key in self.window:
            self.freq[key] += 1
            self.window.move_to_end(key)
            return True
        if key in self.main:
            self.freq[key] += 1
            self.prefetched.pop(key, None)   # a hit graduates a prefetch
            self._push(key)
            return True
        return False

    def _admit_to_main(self, k) -> None:
        """Frequency-gated admission into the protected region."""
        if k in self.main:
            self._push(k)
            return
        if len(self.main) >= self.main_cap:
            if self.prefetched:              # stale prefetches go first
                victim, _ = self.prefetched.popitem(last=False)
            else:
                v = self._peek(self._mheap, self.main)
                if v is None or v[0] > self.freq[k]:
                    return                   # gate closed: drop k
                victim = v[1]
            self.main.discard(victim)
            self.prefetched.pop(victim, None)
        self.main.add(k)
        self._push(k)

    def admit(self, exact_keys, neighborhood, now) -> None:
        for k in exact_keys:
            self.freq[k] += 1
            if k in self.window:
                self.window.move_to_end(k)
                continue
            if k in self.main:
                self._push(k)
                continue
            self.window[k] = True            # recency-friendly entry
            if len(self.window) > self.window_size:
                evictee, _ = self.window.popitem(last=False)
                self._admit_to_main(evictee)
        if neighborhood:
            items = (sorted(neighborhood.items(), key=lambda kv: -kv[1])[: self.prefetch_m]
                     if isinstance(neighborhood, dict) else
                     [(k, 1.0) for k in list(neighborhood)[: self.prefetch_m]])
            for k, _ in items:
                if k in self.main or k in self.window:
                    continue
                if (len(self.prefetched) >= self.prefetch_budget
                        or len(self.main) >= self.main_cap):
                    if not self.prefetched:
                        break                # never displace proven residents
                    victim, _ = self.prefetched.popitem(last=False)
                    self.main.discard(victim)
                self.main.add(k)
                self.prefetched[k] = True    # newest prefetch = last in LRU order
                self._push(k)
        self._ops += 1
        if self._ops % self.age_every == 0:
            for k in list(self.freq):
                self.freq[k] //= 2
                if self.freq[k] == 0 and k not in self.main and k not in self.window:
                    del self.freq[k]
            self._mheap = []
            for k in self.main:
                self._push(k)
