"""Hybrid cache policy: frequency-gated admission + bounded semantic prefetch.

exp4 showed no single policy wins every workload: ghost-frequency admission
(TinyLFU-family) wins skewed/zipf traffic, bounded semantic prefetch wins
drifting traffic, and everything ties on uniform. This policy combines the
two mechanisms and aims to match-or-beat the best specialist in EVERY column:

  - eviction: lowest ghost frequency among residents (aged periodically)
  - exact results: always counted in ghost frequencies; admitted when they
    beat the coldest resident (classic frequency-gated admission)
  - prefetch: only the top-`prefetch_m` neighbors by deposited energy are
    candidates, and each is admitted only against a BUDGET of the cache
    (at most `prefetch_frac` of capacity may hold never-requested items),
    so prefetch can help on drift but cannot pollute under skew.

Tuning protocol: knobs are chosen on seed 0 only and validated untouched on
seeds 1-2 (exp9), so the win cannot be a tuning artifact.
"""

from __future__ import annotations

from collections import defaultdict

from .thermo_cache import BaseCache


class HybridCache(BaseCache):
    def __init__(self, capacity: int, prefetch_m: int = 50,
                 prefetch_frac: float = 0.25, age_every: int = 2000):
        super().__init__(capacity)
        self.prefetch_m = prefetch_m
        self.prefetch_budget = max(1, int(prefetch_frac * capacity))
        self.age_every = age_every
        self.freq: dict = defaultdict(int)   # ghost frequencies (all keys)
        self.store: set = set()
        self.prefetched: set = set()         # residents never yet requested
        self._ops = 0

    def contains(self, key, now) -> bool:
        if key in self.store:
            self.freq[key] += 1
            self.prefetched.discard(key)     # a hit graduates a prefetch
            return True
        return False

    def _evict_one(self, incoming_freq: int) -> bool:
        """Evict the coldest resident if the incoming key deserves the slot.
        Prefetched-but-never-used residents are first in line."""
        pool = self.prefetched if self.prefetched else self.store
        victim = min(pool, key=lambda x: self.freq[x])
        if pool is self.store and self.freq[victim] > incoming_freq:
            return False                     # admission denied
        self.store.discard(victim)
        self.prefetched.discard(victim)
        return True

    def admit(self, exact_keys, neighborhood, now) -> None:
        for k in exact_keys:
            self.freq[k] += 1
            if k in self.store:
                continue
            if len(self.store) >= self.capacity and not self._evict_one(self.freq[k]):
                continue
            self.store.add(k)
        if neighborhood:
            items = (sorted(neighborhood.items(), key=lambda kv: -kv[1])[: self.prefetch_m]
                     if isinstance(neighborhood, dict) else
                     [(k, 1.0) for k in list(neighborhood)[: self.prefetch_m]])
            for k, _ in items:
                if k in self.store:
                    continue
                if len(self.prefetched) >= self.prefetch_budget:
                    break                    # prefetch may not flood the cache
                if len(self.store) >= self.capacity:
                    # Prefetch only displaces other unproven prefetches.
                    if not self.prefetched:
                        break
                    victim = min(self.prefetched, key=lambda x: self.freq[x])
                    self.store.discard(victim)
                    self.prefetched.discard(victim)
                self.store.add(k)
                self.prefetched.add(k)
        self._ops += 1
        if self._ops % self.age_every == 0:  # aging keeps ghosts adaptive
            for k in list(self.freq):
                self.freq[k] //= 2
                if self.freq[k] == 0 and k not in self.store:
                    del self.freq[k]
