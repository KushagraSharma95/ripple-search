# Where Search Is Actually Winnable (2026)

This document answers the question the ripple study raised: if the
energy-ripple hypothesis is dead, what WOULD win? It synthesizes six
research passes over the 2024–2026 record — competition leaderboards,
primary papers, vendor engineering posts — plus this repo's own
measurements. Every claim carries its source; self-reported numbers are
labeled as such.

The doctrine first, because it explains every verdict below: **search has
never been won by beating everything — it is won by beating one constraint
regime the incumbents are structurally weak in.** DiskANN won
billion-scale-on-one-SSD. HNSW won in-memory recall/QPS. ColBERT won
quality-per-FLOP at reranking. Seismic (a small academic team, 2024-25) won
sparse retrieval 2.6–21× over engineered incumbents — proof the pattern
still works. Our own study is a miniature of the same law: isotropic
diffusion lost to directed navigation ~10× on work counters at equal
recall, because mechanism beats metaphor.

## Closed fronts — do not attack

- **Core graph + quantization kernels.** NeurIPS'23 Big-ANN was the last
  official competition; the post-competition leaderboards are now held by
  Pinecone/Zilliz/Baidu (filtered: 85,491 QPS@90% recall; OOD: 38,088 QPS)
  [arXiv:2409.17424, big-ann-benchmarks.com]. Quantization is a vendor
  arms race (RaBitQ/BBQ/SOAR lineages). Wins here are absorbed by
  incumbents within 12–24 months.
- **GPU ANN.** CAGRA/cuVS ships inside Faiss (Meta production), Milvus,
  OpenSearch: build 2.2–27× faster than HNSW, batch throughput 33–77×
  [ICDE'24; Meta 2025]. Competing head-on means competing with NVIDIA's
  hardware-software co-design. Ride it; don't race it.
- **General-purpose embedders on BEIR/MTEB.** Saturated and
  contamination-suspect: top MTEB retrieval models lose 2.7 points to mere
  paraphrasing, and the best BEIR embedder collapses from 59.0 to 18.3
  nDCG@10 on BRIGHT's reasoning queries. The leaderboard is no longer
  measuring progress.
- **Generative retrieval (DSI/NCI).** At 8.8M passages its gains vanish
  when compute-matched; no production deployment found 2025–26. The
  field's own energy-ripple.

## Hype — measured and found wanting

- **Neuromorphic hardware for search.** Zero recall-matched
  neuromorphic-vs-HNSW curves have ever been published. The one Loihi
  k-NN result (Pohoiki Springs, 2020: 3.03 ms, 366 QPS, 1M points) was
  never compared at matched recall and had no 2024–26 follow-up. SpiNNaker
  2 and NorthPole publish no retrieval benchmarks at all. Analog CAMs are
  real physics but 5–10 years from product (2–4 bits/cell, ≤256×256
  arrays, match-line crosstalk). The one near-memory bright spot — UPMEM
  PIM, 4.3× QPS vs CPU Faiss at billion scale [UpANNS, SC'25] — ended
  with UPMEM liquidated and sold in 2025. Verdict: for search, buy
  commodity hardware and spend the cleverness in software.
- **Physics/nature metaphors as engines.** This repo's own verdicts, plus
  the absence of ANY bio-inspired method on a big-ann leaderboard.

## What biology actually contributes (it's not the index)

The bio-inspired survivors all live ABOVE the index, in memory
organization — which is a genuinely open layer:

- **FlyHash / fly Bloom filter** (Science 2017; PNAS 2018): sparse
  expansion beats classical LSH at equal bits and gives cheap streaming
  *novelty detection* — a niche win, never a server-scale ANN win.
- **Hyperdimensional computing**: 169–314× energy wins on edge/FPGA
  tasks; nothing at server scale. Edge-only.
- **Modern Hopfield networks** (2020): exponential-capacity associative
  memory whose update rule IS transformer attention — biology's associative
  search already won, and it's called attention.
- **Kanerva SDM revived inside LLMs**: Larimar (ICML'24) uses an SDM-style
  external memory for 4–10× faster LLM knowledge editing. **HippoRAG**
  (NeurIPS'24) uses hippocampal pattern-separation ideas over a knowledge
  graph: up to +20% multi-hop QA at 10–20× lower cost than iterative-LLM
  retrieval; HippoRAG 2 holds R@5 74.7 on MuSiQue vs 69.7 for the best
  dense embedder. This line — biologically structured MEMORY over
  conventional indexes — is the one place "neuromorphic" ideas are
  currently earning measured wins.

## The three open fronts, ranked by evidence

### 1. Per-query adaptive control with declarative targets (most tractable)

Indexes run one fixed budget for every query; query difficulty varies
wildly. The 2025–26 record shows the biggest replicated margins anywhere
in the stack: **DARTH** (SIGMOD'26) hits declared recall with avg 6.8×
(up to 14.6×) faster HNSW search and 41.8× on IVF, within 5% of
oracle-optimal stopping; LAET (up to 7.1×) and Ada-ef (4× online, 50×
less offline compute) replicate the direction independently. Nobody yet
unifies termination + entry-point routing + reranking cascades under one
declarative contract ("95% recall at minimum cost"), handles OOD queries,
or targets the ≥0.99-recall regime that standard benchmarks don't even
measure.

**This repo's contribution (exp8/exp8b):** even a 20-line stall-based
heuristic — no learning at all — saves 5–24% of distance computations at
matched recall on Reuters, growing to **22–37% at recall 0.988**, while
tying/losing on a homogeneous synthetic corpus. Two lessons measured:
adaptive termination's profit is proportional to query-difficulty
variance (real corpora have it; uniform synthetic benchmarks hide it),
and the win concentrates exactly in the high-recall regime the
literature under-benchmarks. The naive unbounded-patience variant LOSES
to the classic bound everywhere (kept in the results as a cautionary
negative).

### 2. Fresh + filtered vector search at object-storage prices (biggest prize)

Three measured weaknesses converge and no incumbent covers their
intersection:

- *Filtered*: no method dominates across filter types/selectivities/
  scales [ETH FANNS benchmark 2025]; the two best (UNG 10×, Filtered-
  DiskANN) both fail on large transformer-embedding datasets, and UNG
  breaks under updates.
- *Streaming*: recall degrades under churn (0.966→0.461 over 20 cycles at
  5% turnover unpatched); IP-DiskANN/DGAI/OdinANN leapfrog each other on
  different self-benchmarks; the only shared scoreboard (NeurIPS'23
  streaming runbook, top 0.985) doesn't even measure sustained tail
  latency under churn. The first filtered+streaming system (Oct 2026)
  already concedes single-tenant recall and tail latency.
- *Economics*: object-storage designs (Turbopuffer: $0.02/GB-month vs
  $0.60 replicated SSD, $5/GB RAM) reset the cost base, but freshness —
  ingest-to-searchable latency — is their open wound.

A system that holds declared recall under continuous inserts/deletes WITH
predicates at object-storage cost, scored on the big-ann streaming+filtered
runbooks plus a published churn-tail-latency protocol, has no incumbent to
displace — only gaps to fill. This is the ambitious campaign.

### 3. Reasoning-intensive retrieval — BRIGHT (open quality game)

BM25 scores ~14.5 nDCG@10, the best embedder ~18–38, the best full
pipeline ~46.8 — against a ceiling of 100, on a public leaderboard
(brightbenchmark.github.io) where small teams (DIVER, BGE-Reasoner,
ReasonRank) traded the lead through 2025 with fine-tunes of open models.
LLM-reasoned query rewriting alone is worth +12 points. Multi-hop QA
shows the same: RL-trained iterative retrieval (Search-R1 line) ~30 EM vs
~20 for naive RAG. "Intelligent search" in 2026 means this — reasoning
about what to retrieve — not a smarter index kernel.

## How a win gets recognized (scoring rules)

Public, protocol-fixed, counter-based where possible: big-ann-benchmarks
runbooks (filtered/streaming/OOD), the FANNS unified benchmark, BRIGHT
leaderboard, and for engine-level claims, work counters (distance
computations at matched recall) alongside wall clock — this repo's own
discipline. Vendor-vs-vendor benchmark feuds (stale-version allegations,
128-dim datasets standing in for 768–3072-dim production reality) are
exactly why counter-based, protocol-pinned results are the credible
currency.

## The campaign this repo is running

1. **exp8/exp8b (done):** verified the adaptive-termination headroom with
   a learning-free heuristic; negative result on naive patience published.
2. **exp9 (running):** one cache policy to match every specialist across
   zipf/drifting/uniform workloads — tuned on one seed, validated on
   held-out seeds.
3. **Next:** a learned stop-predictor (features: top-k stability, score
   gaps, frontier pressure) targeting declared-recall contracts at ≥0.99
   recall, evaluated on big-ann datasets against DARTH's published
   protocol; then the freshness axis (churn runbook with tail-latency
   reporting) where the scoreboard is softest.
