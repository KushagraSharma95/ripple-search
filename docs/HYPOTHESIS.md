# The Hypothesis Under Test

Source: a speculative architectural concept for an "Emergent Ripple Engine" /
"thermodynamic search database." To make it testable, the prose is pinned down
into numbered, falsifiable claims. Each experiment in `experiments/` targets
one claim; verdicts live in the top-level `README.md`.

## C1 — Flat scaling in N

> "As data grows (N): traditional engines slow down linearly or
> quadratically; the ripple engine's search speed stays flat because energy
> decay restricts the radius."

Two testable parts:
- **C1a**: ripple query cost stays ~flat as N grows, *at unchanged result
  quality* (recall must be reported, otherwise "flat" is trivially achievable
  by returning garbage).
- **C1b**: traditional engines slow down "linearly or quadratically" with N.
  (Note: this premise is itself suspect — B-trees, hash indexes and inverted
  indexes are sublinear by design. The experiment measures real engines.)

Tested by: `exp1_scaling_n.py`, `exp1b_recall_matched.py`,
`exp1c_fts5_scaling.py`, `exp5_real_text.py`.

## C2 — Dimensionality sharpens pruning

> "As dimensions grow (D): the curse of dimensionality crashes traditional
> indexes, but search becomes MORE accurate because geometric pruning angles
> get sharper. The triangle inequality math becomes far more aggressive. A
> single corner check can instantly rule out a massive percentage of the
> database."

Testable part: triangle-inequality pruning effectiveness (fraction of the
database that can be skipped) should INCREASE with D; the ripple engine's
accuracy at fixed energy budget should not degrade with D.

Tested by: `exp2_dimensionality.py` (instrumented VP-tree distance counts,
relative contrast, plus the hyperdimensional-computing counterpoint for random
binary codes).

## C3 — Hebbian self-optimization

> "Every search leaves a residual trace like an ant pheromone trail. If nodes
> are frequently activated by the same query ripple, their connection
> strengthens. The network organically self-optimizes, burning highly traveled
> paths into permanent shortcuts. As more query cycles enter the system the
> search becomes more intelligent, faster, and more precise."

Testable part: on workloads with locality (Zipf topic skew, drifting "train of
thought"), an adaptive engine should visit fewer nodes and/or achieve higher
recall over time than an identical frozen twin. Side condition worth
measuring: what happens to out-of-distribution queries after adaptation.

Tested by: `exp3_hebbian_adaptation.py`.

## C4 — Thermodynamic caching

> "The high energy liquefies the entire local neighborhood of nodes and pumps
> the activated cluster into cache. Eviction is natural dissipation — data
> organically falls out when its energy hits zero. This proactive,
> topological-cluster caching beats reactive single-key LRU/LFU policies, and
> the database stays one step ahead of the user."

Testable part: at equal capacity, temperature-decay eviction + neighborhood
admission should beat LRU and LFU hit rates on localized workloads. The
honest ablation separates the two bundled mechanisms (admission policy vs
eviction policy), and includes a no-locality workload where prefetching
should backfire (cache pollution).

Tested by: `exp4_thermo_cache.py`.

## C5 — Log-time consolidation

> "As data ages, the system automatically merges old nodes into dense,
> singular historical macro-nodes. A query ripple hits a single, highly stable
> historical hub instead of billions of scattered events. Time effectively
> self-indexes the database, keeping the active search field perpetually small
> and lightning-fast."

Testable part: consolidation should reduce per-query work substantially. The
hypothesis is silent about the price; the experiment measures recall on
recent-targeting AND archive-targeting queries before/after consolidation.

Tested by: `exp6_temporal_consolidation.py`.

## C6 — Faster than Teradata / Oracle

The user's direct question: is this faster than search in Teradata or Oracle
databases?

Oracle IS tested directly: Oracle Database Free (26ai) runs in a container
under the Free Use Terms and Conditions (read in full — no benchmark-
disclosure clause; see `docs/METHODOLOGY.md`), and `exp7_oracle_text.py`
benchmarks its Oracle Text CONTEXT inverted index on the same corpus and
queries as every other engine. Teradata cannot be run honestly here
(registration-gated VM distribution) and has no native full-text engine
comparable to Oracle Text — text workloads are typically handled via LIKE
scans, external indexing, or UDFs; its identity is hash-distributed
relational access, which is ~O(1) in N for primary-index lookups. For the
text-search claim its same-architecture open stand-ins are SQLite FTS5 (C
inverted index + BM25) and PostgreSQL tsvector/GIN.

Tested by: `exp5_real_text.py`, `exp5b_short_queries.py`,
`exp7_oracle_text.py` on Reuters-21578; N-scaling of a real inverted index
in `exp1c_fts5_scaling.py`.
