# Methodology

## Principles

Following the ann-benchmarks discipline (Aumüller, Bernhardsson, Faithfull,
*ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor
Algorithms*, Information Systems 2020):

1. **Latency is meaningless without quality.** Every approximate engine
   reports recall@10 against exact brute-force ground truth on the identical
   query set. Text engines additionally report same-category precision@10 on
   labeled data (Reuters-21578 topic labels).
2. **Build time and query time are reported separately.** A structure may buy
   query speed with expensive construction; both numbers are visible.
3. **Single-threaded query timing**, `time.perf_counter`, GC disabled in the
   timed region, warmup queries excluded, p50/p95/mean over hundreds of
   queries.
4. **Language-independent work counters.** The ripple engine is Python/numpy;
   FTS5, hnswlib and PostgreSQL are C/C++. Wall-clock alone would confound
   language overhead with algorithmic merit, so the ripple engine and the
   instrumented VP-tree also report nodes visited and true distance
   computations per query. Claims about *scaling shape* are evaluated on
   counters; claims about *absolute speed* on wall clock.
5. **No strawmen.** The ripple engine's parameters are calibrated by grid
   search on a held-out validation query set (`exp0_calibrate.py`) before any
   benchmark, the same courtesy HNSW's `ef` gets. Adaptive mechanisms are
   ablated with identical frozen twins (same build, same seeds).
6. **Determinism.** Fixed seeds for corpus generation, graph build, and query
   streams. Calibration uses a query seed disjoint from all experiments.

## Oracle: benchmarked directly. Teradata: proxied. Here is why.

- **Oracle — run for real.** Oracle Database Free (26ai, `gvenzl/oracle-free`
  container image) ships under the *Free Use Terms and Conditions* (FUTC).
  The full license embedded in the image at `$ORACLE_HOME/LICENSE` (last
  updated 8 Oct 2018) was read before publishing: it grants the right to
  "internally use the unmodified Programs for the purposes of developing,
  testing, prototyping and demonstrating your applications" and — unlike
  Oracle's OTN developer license — contains **no benchmark-disclosure
  ("DeWitt") clause**. `exp7_oracle_text.py` therefore benchmarks the actual
  Oracle Text CONTEXT index (an inverted index with CONTAINS/SCORE ranking)
  on the same corpus and queries as every other text engine. Latency includes
  a localhost Docker-bridge TCP round trip; FTS5 is in-process and PostgreSQL
  uses a unix socket — absolute milliseconds across engines carry that caveat,
  scaling shapes do not.
- **Teradata — cannot be run honestly here.** Teradata Vantage Express is
  registration-gated and distributed as a multi-GB VM image; ClearScape
  Analytics Experience is a hosted signup service. More fundamentally,
  Teradata has no native inverted-index full-text engine comparable to Oracle
  Text — its strength is hash-distributed parallel relational access
  (primary-index lookups are single-AMP O(1)-ish hash probes, not scans). For
  the text-search claim, SQLite FTS5 and PostgreSQL GIN+tsvector are the
  same-architecture open proxies.
- One scaling fact from vendor documentation worth stating up front: inverted
  index postings-list lookup is ~O(terms + log vocab), B-tree access is
  O(log N), and hash access is ~O(1). "Linear or quadratic in N" is not how
  any of these engines execute an indexed search.

## Datasets

- **Synthetic**: Gaussian-mixture topic corpora on the unit sphere
  (`ripple_search/workloads.py`) with controllable N, D, topic count and
  spread. Used where a controlled sweep is required (N-scaling, D-sweep,
  adaptation, caching, consolidation). Both clustered (realistic) and i.i.d.
  (worst-case) regimes appear in the dimensionality experiment.
- **Real text**: Reuters-21578 (10,788 newswire documents, 90 topic labels),
  a corpus that has been an IR benchmark since 1990. Embedding side: TF-IDF
  (sublinear tf, min_df=3) -> TruncatedSVD (LSA, 256 dims) -> unit norm.
  Text side: raw text to FTS5/PostgreSQL with their native tokenization.
- Queries for real text are the lead 40 words of 300 sampled documents; the
  query document itself is excluded from quality scoring.

## Workloads (locality models)

- `uniform` — no exploitable locality (adversarial for caching/adaptation)
- `zipf` — few topics dominate (favorable locality)
- `drifting` — the hypothesis's own scenario: a "train of thought" random
  walk where consecutive queries hit semantically adjacent topics

## Known limitations

- Single machine, in-memory, 4 vCPU container; no disk-resident or
  distributed behavior is measured.
- Corpus sizes up to 100k documents — small by enterprise standards, but the
  *scaling shapes* (the claims under test) are visible well below that.
- The ripple engine is a faithful but single implementation of prose; other
  parameterizations of "energy physics" exist. The calibration sweep explores
  48 configurations to reduce the strawman risk.
- Latency comparisons between text engines (FTS5, PostgreSQL) and vector
  engines (ripple, HNSW, brute force) cross modalities: they answer "which
  system answers a topical query faster at what quality", not "which
  algorithm is better at the same job". Both quality metrics are reported.
