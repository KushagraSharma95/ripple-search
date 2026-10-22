"""Thermodynamic cache and classical baselines (LRU, LFU, prefetch ablations).

The hypothesis: when a search terminates, the ENTIRE activated neighborhood is
pumped into cache with temperature proportional to deposited energy; cached
items radiate (stay warm while related queries arrive) and cool exponentially;
eviction is "natural dissipation" — coldest items leave first.

To separate the two mechanisms the hypothesis bundles together, we ablate:
  - admission policy: exact-items-only vs activated-neighborhood (prefetch)
  - eviction policy: LRU / LFU bookkeeping vs decayed-temperature

So the grid is: LRU, LFU, LRU+prefetch, Thermo (exact), Thermo+prefetch.
"""

from __future__ import annotations

import math
from collections import OrderedDict, defaultdict


class BaseCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def probe(self, keys, now: float) -> int:
        """Count hits for this query's needed keys (before admission)."""
        h = sum(1 for k in keys if self.contains(k, now))
        self.hits += h
        self.misses += len(keys) - h
        return h

    def contains(self, key, now) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def admit(self, exact_keys, neighborhood, now) -> None:  # pragma: no cover
        raise NotImplementedError


class LRUCache(BaseCache):
    def __init__(self, capacity: int, prefetch: bool = False):
        super().__init__(capacity)
        self.prefetch = prefetch
        self.store: OrderedDict = OrderedDict()

    def contains(self, key, now) -> bool:
        if key in self.store:
            self.store.move_to_end(key)
            return True
        return False

    def admit(self, exact_keys, neighborhood, now) -> None:
        keys = list(exact_keys) + (list(neighborhood) if self.prefetch else [])
        for k in keys:
            if k in self.store:
                self.store.move_to_end(k)
            else:
                self.store[k] = True
                if len(self.store) > self.capacity:
                    self.store.popitem(last=False)


class LFUPlainCache(BaseCache):
    """Textbook in-cache LFU: frequency tracked only while resident, counts
    deleted on eviction, every new item admitted (evicting the lowest-
    frequency resident). The contrast class for LFUCache below, which keeps
    ghost frequencies for non-resident keys and gates admission on them —
    making it a TinyLFU-family policy, not 'plain LFU'."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.freq: dict = {}

    def contains(self, key, now) -> bool:
        if key in self.freq:
            self.freq[key] += 1
            return True
        return False

    def admit(self, exact_keys, neighborhood, now) -> None:
        for k in exact_keys:
            if k in self.freq:
                self.freq[k] += 1
                continue
            if len(self.freq) >= self.capacity:
                victim = min(self.freq, key=self.freq.get)
                del self.freq[victim]
            self.freq[k] = 1


class LFUCache(BaseCache):
    """LFU with GHOST frequencies and frequency-gated admission (keeps counts
    for non-resident keys; refuses admission when the coldest resident has a
    higher count). Strictly stronger than plain LFU — this is TinyLFU-family
    behavior, and it is labeled `lfu_ghost_admission` in exp4's results."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.freq: dict = defaultdict(int)
        self.store: set = set()
        self._ops = 0

    def contains(self, key, now) -> bool:
        if key in self.store:
            self.freq[key] += 1
            return True
        return False

    def admit(self, exact_keys, neighborhood, now) -> None:
        for k in exact_keys:
            self.freq[k] += 1
            if k in self.store:
                continue
            if len(self.store) >= self.capacity:
                victim = min(self.store, key=lambda x: self.freq[x])
                if self.freq[victim] > self.freq[k]:
                    continue
                self.store.discard(victim)
            self.store.add(k)
        self._ops += 1
        if self._ops % 2000 == 0:  # aging
            for k in list(self.freq):
                self.freq[k] //= 2


class ThermoCache(BaseCache):
    """Temperature cache: admit with heat, decay exponentially, evict coldest.

    temperature(t) = T * exp(-(t - last_heated) / tau)
    - probe hit re-heats the item (energy deposit on access)
    - admission heats exact results with `hit_heat`, neighborhood items with
      the ripple energy they received (scaled)
    - eviction removes the coldest items when over capacity
    """

    def __init__(self, capacity: int, tau: float = 200.0, prefetch: bool = True,
                 hit_heat: float = 1.0, neighbor_heat_scale: float = 0.5):
        super().__init__(capacity)
        self.tau = tau
        self.prefetch = prefetch
        self.hit_heat = hit_heat
        self.neighbor_heat_scale = neighbor_heat_scale
        self.temp: dict = {}
        self.last: dict = {}

    def _temperature(self, key, now) -> float:
        return self.temp[key] * math.exp(-(now - self.last[key]) / self.tau)

    def _heat(self, key, amount, now) -> None:
        cur = self._temperature(key, now) if key in self.temp else 0.0
        self.temp[key] = cur + amount
        self.last[key] = now

    def contains(self, key, now) -> bool:
        if key in self.temp:
            self._heat(key, self.hit_heat, now)
            return True
        return False

    def admit(self, exact_keys, neighborhood, now) -> None:
        for k in exact_keys:
            self._heat(k, self.hit_heat, now)
        if self.prefetch:
            if isinstance(neighborhood, dict):
                for k, e in neighborhood.items():
                    self._heat(k, self.neighbor_heat_scale * e, now)
            else:
                for k in neighborhood:
                    self._heat(k, self.neighbor_heat_scale * self.hit_heat, now)
        if len(self.temp) > self.capacity:
            # Evict the coldest overflow ("dissipation"). Batch eviction keeps
            # this O(n log n) only when actually over capacity.
            items = sorted(self.temp, key=lambda k: self._temperature(k, now))
            for k in items[: len(self.temp) - self.capacity]:
                del self.temp[k]
                del self.last[k]
