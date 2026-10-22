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
                 age_every: int = 2000, adaptive: bool = False,
                 adapt_epoch: int = 250, adapt_step: float = 0.06):
        super().__init__(capacity)
        self.prefetch_frac = prefetch_frac
        self._resize(window_frac)
        self.prefetch_m = prefetch_m
        self.age_every = age_every
        self.adaptive = adaptive
        self.adapt_epoch = adapt_epoch
        self.adapt_step = adapt_step
        self._dir = 1.0
        self._prev_rate = None
        self._last_hits = 0
        self._last_probes = 0
        self.freq: dict = defaultdict(int)   # ghost frequencies (all keys)
        self.window: OrderedDict = OrderedDict()
        self.main: set = set()
        self.prefetched: OrderedDict = OrderedDict()  # unproven prefetches, LRU order
        self._mheap: list = []               # lazy (freq, seq, key) over main
        self._seq = 0
        self._ops = 0

    # -------------------------------------------------------- adaptive sizing
    def _resize(self, window_frac: float) -> None:
        self.window_frac = window_frac
        self.window_size = max(1, int(window_frac * self.capacity))
        self.main_cap = max(1, self.capacity - self.window_size)
        self.prefetch_budget = max(1, int(self.prefetch_frac * self.main_cap))

    def _enforce_sizes(self) -> None:
        # Window shrank: demote LRU evictees through the frequency gate.
        while len(self.window) > self.window_size:
            evictee, _ = self.window.popitem(last=False)
            self._admit_to_main(evictee)
        # Main shrank: drop stale prefetches first, then coldest residents.
        while len(self.main) > self.main_cap:
            if self.prefetched:
                victim, _ = self.prefetched.popitem(last=False)
            else:
                v = self._peek(self._mheap, self.main)
                if v is None:
                    break
                victim = v[1]
            self.main.discard(victim)
            self.prefetched.pop(victim, None)
        while len(self.prefetched) > self.prefetch_budget:
            victim, _ = self.prefetched.popitem(last=False)
            self.main.discard(victim)

    def _adapt(self) -> None:
        # Hill-climb window_frac on epoch hit rate (Caffeine-style): keep
        # moving while the rate improves, reverse when it drops.
        probes = (self.hits + self.misses) - self._last_probes
        rate = (self.hits - self._last_hits) / max(1, probes)
        self._last_hits = self.hits
        self._last_probes = self.hits + self.misses
        if self._prev_rate is not None and rate < self._prev_rate:
            self._dir = -self._dir
        self._prev_rate = rate
        new = min(0.8, max(0.02, self.window_frac + self._dir * self.adapt_step))
        if new != self.window_frac:
            self._resize(new)
            self._enforce_sizes()

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
        if self.adaptive and self._ops % self.adapt_epoch == 0:
            self._adapt()
        if self._ops % self.age_every == 0:
            for k in list(self.freq):
                self.freq[k] //= 2
                if self.freq[k] == 0 and k not in self.main and k not in self.window:
                    del self.freq[k]
            self._mheap = []
            for k in self.main:
                self._push(k)
