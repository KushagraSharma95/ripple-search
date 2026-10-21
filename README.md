# Ripple Search POC — an honest test of the "thermodynamic ripple engine"

An AI conversation produced an architecture for an "Emergent Ripple Engine":
documents as nodes at hypercube corners, queries as kinetic-energy ripples
with decay, Hebbian path-strengthening, thermodynamic caching, and log-time
consolidation — with claims that it beats traditional databases and that
**more data, more dimensions and more time make it faster and more accurate**.

This repository implements that architecture faithfully (`ripple_search/`),
calibrates it fairly (176 configurations, Pareto-front selection), and tests
every claim against exact ground truth and real engines: brute-force vector
scan, **HNSW** (hnswlib), an instrumented **VP-tree**, **SQLite FTS5**,
**PostgreSQL GIN/tsvector**, and a real **Oracle Database Free (26ai) with
Oracle Text** running in a container. Six independent literature passes over
primary sources were done *before* benchmarking (`docs/FINDINGS.md`).

The claims, as falsifiable statements: `docs/HYPOTHESIS.md`.
The rules of the game: `docs/METHODOLOGY.md`.

## Verdicts

| # | Claim | Verdict | Decisive measurement |
|---|-------|---------|----------------------|
| C1a | "As data grows, search speed stays flat" | **Refuted** | At calibrated quality (recall ≥ 0.87), p50 latency grows 12 ms → 170 ms as N goes 2k → 100k, and recall still slides 0.99 → 0.87. Recall-matched at N=100k, ripple is **487× slower** than HNSW (169.8 ms vs 0.35 ms). [exp1, exp1b] |
| C1b | "Traditional engines slow down linearly or quadratically with N" | **Refuted** | Inverted indexes, B-trees and hash access are sublinear by construction (postings ~O(terms+log V), B-tree O(log N), hash ~O(1)); Oracle Text needed 3.6 s to index 10,788 docs and answers in 12 ms. The premise misdescribes how databases work. [exp7, FINDINGS] |
| C2 | "More dimensions = sharper pruning and more accuracy" | **Refuted** | Triangle-inequality pruning (VP-tree) touches 0.2% of the DB at d=2, 96% at d=16, **100% (= brute force) at d≥32**. Relative contrast collapses 33,113 → 0.18. The ripple's own recall at fixed energy falls 1.00 → 0.02 from d=32 to d=512. Exactly the curse of dimensionality the literature proved in 1998–1999. [exp2] |
| C3 | "Hebbian trails make it faster and more precise over time" | **Refuted** | After 4,000 queries on favorable (Zipf/drifting) workloads: nodes visited went **up ~5%** (10,474 → 10,979), recall flat (0.965 → 0.965 Zipf; +0.013 drifting). The mechanism self-defeats: strengthened edges slow energy decay, so the ripple spreads *wider*. No OOD damage either — it simply does ~nothing. [exp3] |
| C4 | "Thermodynamic caching beats reactive LRU/LFU" | **Refuted as stated** | Zipf: plain LFU wins (0.236 vs thermo 0.202). Drifting ("train of thought", the hypothesis's own scenario): **LRU + bounded prefetch wins** (0.159 vs thermo 0.143). Admitting the whole activated neighborhood — the actual claim — is the *worst* policy measured (0.096–0.099): textbook cache pollution. Decayed-temperature eviction is mathematically the LRFU family (1990s), and Oracle's own buffer cache already uses touch-count heat. [exp4, FINDINGS] |
| C5 | "Log-time consolidation keeps the search field small and fast" | **Mixed** | Real: 50k nodes → 9.1k (5.5× compression), per-query work falls 7.1× (18.6k → 2.6k nodes). The bill the hypothesis never mentions: recall drops 0.90 → 0.80 on *recent*-targeting queries and 0.92 → 0.85 on archival ones. A real speed/recall trade, not a free lunch — and existing systems get it via time-partitioned indexes without the recall loss on recent data. [exp6] |
| C6 | "Faster than search in Teradata or Oracle databases" | **Refuted** | Measured against the real thing: Oracle Text (CONTEXT index, ACCUM ranking) answers the same 300 Reuters queries at **p50 ≈ 12 ms** (7.7 ms for 10-word queries) where the ripple engine needs **48.9 ms** — and a plain brute-force numpy scan needs 0.58 ms. Teradata cannot be raced (registration-gated VM, and it has **no native full-text engine** — its strength is O(1)-ish hash primary-index access, which no diffusion process will beat). [exp5, exp7] |

## The autopsy, in one paragraph

The hypothesis is five known mechanisms wearing a physics costume, plus two
claims that are provably backwards. Content-addressable memory on hypercube
corners with activation spreading is **Kanerva's Sparse Distributed Memory
(1988)**; query-driven self-strengthening paths are **ant-colony routing /
spreading activation / adaptive indexing** (all with measured, modest, known
trade-offs); temperature-decay eviction is the **LRFU family (1990s)** — and
Oracle's buffer cache has used touch-count "heat" for decades; neighborhood
prefetching is **semantic caching/correlated prefetch**, with cache pollution
as its documented failure mode (measured here: it's the worst policy in the
grid). The two backwards claims: high dimensionality makes geometric pruning
*vacuous*, not sharper (Beyer et al. 1999; Weber et al. 1998; measured in
exp2 at 100% of the database touched from d=32), and "flat speed as N grows"
is bought only by letting recall collapse. The deepest measured finding:
**energy diffusion is the wrong physics for search** — it spreads effort
isotropically around entry points, while greedy graph navigation (HNSW)
*descends* toward the query; at equal recall that difference is worth three
orders of magnitude. What *is* salvageable — proximity-graph search, bounded
semantic prefetch on drifting workloads, time-partitioned compaction — is
precisely what production systems already do, minus the metaphor.

## Experiments

Plots are in `results/plots/`, raw numbers in `results/*.json`, and each
experiment file's docstring states the claim it tests.

### exp0 / exp0b — calibration (no strawmen)
Grid search over 176 ripple configurations on a validation query set (seed
disjoint from all experiments). The full recall/work Pareto front is
published. Chosen operating points: *budget* (≤5% of corpus visited → recall
0.29) and *quality* (recall 0.97 at N=20k → visits ~54% of the corpus). That
front **is** the finding: there is no configuration that is both fast and
accurate.

### exp1 / exp1b — scaling in N
![scaling](results/plots/exp1_scaling_n.png)
Latency and recall vs corpus size, all engines at fixed settings; then an
HNSW `ef` sweep at N=100k for a recall-matched comparison. Ripple quality
config: work grows ~N^0.7 (not flat), recall falls anyway. HNSW at matched
recall 0.87: 0.35 ms vs ripple 169.8 ms.

### exp2 — dimensionality
![dimensionality](results/plots/exp2_dimensionality.png)
The instrumented VP-tree counts exactly how much the triangle inequality
prunes, per dimensionality, on clustered and i.i.d. data. Pruning dies by
d≈32; distance concentration tracks Beyer et al.; ripple and HNSW recall
both degrade at fixed budgets. One nuance the hypothesis accidentally
touches: for **random binary codes in Hamming space**, high-D concentration
is a *feature* (quasi-orthogonality — Kanerva's hyperdimensional computing).
It just doesn't transfer to metric nearest-neighbor on correlated real data,
which is what a search engine does. Measured in the same run:
`hamming_sep_std_frac` shrinks with D while metric contrast collapses.

### exp3 — Hebbian adaptation
![hebbian](results/plots/exp3_hebbian.png)
Identical twin engines (same build, same seeds), one adaptive, one frozen,
4,000-query streams. Windows of 250 queries. Adaptation changes work by +5%
(wrong direction) and recall by +0.000–0.013. OOD recall after adaptation:
0.963 vs frozen 0.958 — no damage, no benefit.

### exp4 — thermodynamic cache
![cache](results/plots/exp4_cache.png)
Probe-before-admit hit rates on needed top-10 sets, 4,000-query streams,
capacities 250–2000. Six policies ablating admission (exact / top-100-energy
/ full neighborhood) × eviction (LRU / LFU / temperature decay).

### exp5 / exp5b / exp7 — real text, real engines (Reuters-21578)
TF-IDF→LSA embeddings for vector engines; native tokenization for text
engines; 300 lead-text queries; same-category precision@10 as the common
quality metric (self excluded).

| engine | p50 (40-word queries) | p50 (10-word) | category P@10 (40w) |
|---|---|---|---|
| HNSW (ef=64) | **0.08 ms** | — | 0.879 |
| brute-force numpy (exact cosine) | **0.58 ms** | — | 0.879 |
| **Oracle Text CONTEXT (ACCUM)** | 11.5 ms | 7.7 ms | 0.807 |
| SQLite FTS5 (BM25) | 40.1 ms | 7.6 ms | 0.847 |
| ripple (quality config) | 48.9 ms | — | 0.862 |
| PostgreSQL GIN + ts_rank | 55.0 ms | 22.1 ms | 0.778 |

Oracle Text index build: 2.7 s for 10,788 docs. An earlier run using Oracle's
`|` (OR) operator scored catP@10 = 0.36 — because OR ranks by the *maximum*
single-term score; the published numbers use `ACCUM`, Oracle's accumulating
relevance operator, which is the fair reading. The mistake and fix are left
documented here deliberately: misranked operators are exactly how casual
"benchmarks" slander engines.

Notes: 40-term OR/ACCUM queries are pathological for postings-list engines
(every term's postings merged) — exp5b/exp7 therefore also measure the
10-word query shape those engines are built for. Oracle runs in Docker and
is queried over a localhost TCP bridge; FTS5 is in-process; PostgreSQL uses
a unix socket. Absolute cross-engine milliseconds carry that caveat; the
orders of magnitude do not.

### exp6 — temporal consolidation
Before/after consolidating a 50k corpus (90% archive): 5.5× node
compression, 7.1× less work per query, −0.10/−0.07 recall on recent/archival
queries. The one mechanism that does roughly what the prose promises — at a
price the prose never mentions.

## "Can you really benchmark Oracle? What about Teradata?"

**Oracle — yes, legally.** Oracle Database Free ships under the *Free Use
Terms and Conditions*. The license embedded in the container image
(`$ORACLE_HOME/LICENSE`, last updated 8 Oct 2018) grants internal use for
"developing, testing, prototyping and demonstrating" and — unlike Oracle's
OTN developer license — contains **no benchmark-disclosure ("DeWitt")
clause**. It was read in full before publishing these numbers.

**Teradata — no.** Vantage Express is registration-gated and VM-distributed;
ClearScape is a hosted signup service. Architecturally there is also nothing
to race: Teradata has no native inverted-index full-text engine; its identity
is hash-distributed parallel relational access, where a primary-index lookup
is a single-AMP hash probe — effectively O(1) in N, which is *better*
scaling than anything the hypothesis proposes.

## Reproduce

```bash
pip install -r requirements.txt
# Reuters corpus (NLTK data): python -m nltk.downloader reuters
python experiments/run_all.py          # exp1–exp6 (~1 h on 4 vCPUs)
python experiments/exp1b_recall_matched.py
# Oracle (optional): docker run -d --name oracle -e ORACLE_PASSWORD=ripple gvenzl/oracle-free:23
ORACLE_DSN=<container-ip>:1521/FREEPDB1 python experiments/exp7_oracle_text.py
python experiments/make_plots.py
```

Environment for the published numbers: 4 vCPU / 15 GB Linux container,
Python 3.13, numpy 2.5, hnswlib 0.8, PostgreSQL 16.15, SQLite FTS5, Oracle
Database Free 26ai (23.26.3). Query timing single-threaded, GC disabled,
warmed up, p50/p95 over 100–300 queries; builds reported separately; fixed
seeds throughout (see `docs/METHODOLOGY.md`, including known limitations —
single machine, in-memory, N ≤ 100k, single-seed runs).

## Repo layout

```
ripple_search/      the faithful implementation + baselines
  engine.py         energy ripple, corner addressing, Hebbian, consolidation
  thermo_cache.py   temperature cache + LRU/LFU/prefetch ablations
  metric_tree.py    instrumented VP-tree (distance-computation counters)
  baselines.py      brute force, HNSW, FTS5, PostgreSQL FTS
experiments/        exp0–exp7, one claim each (docstrings say which)
results/            raw JSON + plots
docs/               HYPOTHESIS.md · METHODOLOGY.md · FINDINGS.md
```
