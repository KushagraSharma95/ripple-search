# Literature Findings

Produced by six independent research passes over primary sources before any
benchmark was run. Verdicts refer to the claims in `HYPOTHESIS.md`.

## Does high dimensionality sharpen geometric pruning?

The hypothesis is refuted on every testable component. The literature says geometric/metric pruning gets WEAKER, not sharper, as dimensionality grows, and the mechanism is distance concentration: Beyer et al. (ICDT 1999) prove nearest and farthest distances converge (D_max/D_min -> 1), observable from ~10-15 dimensions; since triangle-inequality lower bounds are differences of distances, they shrink to uselessness exactly when distances concentrate (Chavez et al. 2001; TRIM 2025 states this verbatim for modern vector search). The 'sharper diagonal angles' claim inverts reality: angles between high-D vectors concentrate at 90 degrees (Cai, Fan & Jiang, JMLR 2013), and the hypercube diagonal's angle to every axis approaches 90 degrees (cos = 1/sqrt(d)). The 'corner check rules out massive percentages' claim is the precise failure Weber, Schek & Blott (VLDB 1998) proved: above ~10 dimensions all bounding-region indexes degrade to sequential scan because query balls intersect nearly every region, which is why the VA-file abandons pruning for a compressed scan, and why KD-trees lose to brute force past D ~ 20 (scikit-learn docs). The existence of LSH (Indyk & Motwani 1998, invented because exact sub-brute-force search needs exponential-in-d resources) and graph-based ANN (HNSW) is direct historical evidence against the hypothesis. The one legitimate nuance: pruning survives in high AMBIENT dimension when INTRINSIC dimension is low (cover trees, Beygelzimer et al. 2006, cost depends on expansion constant not D), when data is clustered/structured (Beyer et al.'s own caveats; Durrant & Kaban 2009 converse theorem), and decay is slower under L1/fractional norms (Aggarwal et al. 2001) — mitigations, never a reversal. For the POC: benchmark pruning ratio vs dimension on i.i.d. data (expect collapse by D ~ 10-20) and separately on a low-intrinsic-dimension manifold embedded in high D (the only regime where the engine's pruning can look good), and report intrinsic dimension alongside ambient D so apparent wins are attributed correctly.

### REFUTED: The higher the number of dimensions, the sharper the diagonal angles become.

The opposite is true: angles flatten toward 90 degrees, they do not sharpen. Cai, Fan & Jiang (2013, 'Distributions of Angles in Random Packing on Spheres', JMLR 14) prove that as dimension p grows, pairwise angles between random points concentrate around pi/2, and the concentration gets STRONGER with dimension — this is the rigorous basis for the folklore that high-dimensional random vectors are nearly orthogonal. Wegner's lecture notes (arXiv:2101.05841) quantify it: inner products of random unit vectors are O(1/sqrt(d)). For the hypercube specifically, the main diagonal makes an angle with each axis whose cosine is 1/sqrt(d), so the diagonal becomes MORE orthogonal (closer to 90 degrees) to every axis as d grows, not sharper. Angular discrimination between points degrades with dimension; it never sharpens.

Sources:
- https://www.jmlr.org/papers/volume14/cai13a/cai13a.pdf
- https://arxiv.org/pdf/2101.05841
- https://zeroentropy.dev/concepts/orthogonality-concentration/

### REFUTED: The triangle inequality math becomes far more aggressive in high dimensions.

Triangle-inequality pruning power is governed by the spread of the distance distribution, and that spread collapses in high dimensions (distance concentration). Beyer, Goldstein, Ramakrishnan & Shaft (1999, 'When Is Nearest Neighbor Meaningful?', ICDT, LNCS 1540, pp. 217-235) prove that under broad conditions D_max/D_min -> 1 as dimension grows, with the effect empirically visible at as few as 10-15 dimensions. The triangle-inequality lower bound |d(q,p) - d(p,x)| is a DIFFERENCE of two distances; when all pairwise distances become nearly equal, that difference tends to zero and prunes nothing. Chavez, Navarro, Baeza-Yates & Marroquin (2001, 'Searching in Metric Spaces', ACM Computing Surveys 33(3):273-321) formalize this: intrinsic dimensionality rho = mu^2/(2*sigma^2) of the distance histogram, and metric-space pruning degrades as mean grows and variance shrinks — exactly the high-D regime; no exact technique in their survey copes with intrinsic dimension beyond roughly 20. Current work agrees: Song et al. (2025, 'TRIM', arXiv:2508.17828) state that triangle-inequality lower-bound pruning 'becomes less effective in high-dimensional settings' because distance concentration makes the bounds 'too small to be useful', even with optimized landmarks. The inequality remains valid in all dimensions; its BOUNDS become vacuous, the precise opposite of 'more aggressive'.

Sources:
- https://link.springer.com/chapter/10.1007/3-540-49257-7_15
- https://members.loria.fr/MOBerger/Enseignement/Master2/Exposes/beyer.pdf
- http://algorithmics.lsi.upc.edu/docs/practicas/p273-chavez.pdf
- https://arxiv.org/abs/2508.17828

### REFUTED: A single corner check can instantly rule out a massive percentage of the database (geometric pruning rules out more as D grows).

Weber, Schek & Blott (1998, 'A Quantitative Analysis and Performance Study for Similarity-Search Methods in High-Dimensional Spaces', VLDB, pp. 194-205) prove analytically that ALL partitioning/clustering index structures (R*-tree, X-tree, and by their argument any hierarchy of bounding regions) degrade to O(n) — effectively sequential scan — as dimensionality grows: above roughly 10 dimensions a plain sequential scan outperforms them on average. The mechanism: the query ball intersects nearly every partition region, so a bounding-region ('corner') check rules out almost NOTHING. Their VA-file abandons geometric pruning entirely and instead accelerates the unavoidable scan with compressed approximations, beating hierarchical indexes above D ~ 5. KD-trees show the same wall: scikit-learn's official documentation puts efficient KD-tree operation at roughly D < 20, beyond which queries become O(DN) and 'slower than brute force'. Blott & Weber reiterated the analysis at VLDB 2008 ('What's Wrong with High-Dimensional Similarity Search?').

Sources:
- https://www.semanticscholar.org/paper/A-Quantitative-Analysis-and-Performance-Study-for-Weber-Schek/63eaeb0c48175065ffd096aad10aed712c6d7bbb
- http://www.vldb.org/pvldb/vol1/1453861.pdf
- https://scikit-learn.sourceforge.net/stable/modules/neighbors.html
- https://scikit-learn.org/stable/modules/neighbors.html

### REFUTED: Complexity gives the algorithm a sharper scalpel (metric pruning gets SHARPER as dimensionality grows).

The entire post-1998 ANN literature exists BECAUSE exact geometric pruning fails in high D. Indyk & Motwani (1998, 'Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality', STOC; journal version Har-Peled, Indyk & Motwani 2012, Theory of Computing 8(14):321-350) state that exact nearest-neighbor data structures beating brute force require space or query time EXPONENTIAL in the dimension; LSH was invented to escape this by accepting approximation and randomization. Graph-based methods (Malkov & Yashunin, 'Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs', IEEE TPAMI 42(4):824-836) are likewise approximate, give no exactness guarantee, and even they have adversarial instances with empirically linear query time (Indyk et al., NeurIPS 2023, 'Worst-case Performance of Popular Approximate Nearest Neighbor Search Implementations'). If dimensionality sharpened pruning, exact tree indexes would dominate at high D and LSH/HNSW would never have been needed; the opposite happened.

Sources:
- https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/IndykM-curse.pdf
- https://theoryofcomputing.org/articles/v008a014/
- https://proceedings.neurips.cc/paper_files/paper/2023/file/d0ac28b79816b51124fcc804b2496a36-Paper-Conference.pdf

### MIXED: There exists SOME regime where high ambient dimensionality does not destroy (or even helps) pruning/search.

Partially true, but only via LOW INTRINSIC dimension or structure — never via raw ambient dimension sharpening bounds. (1) Intrinsic vs ambient: Beygelzimer, Kakade & Langford (2006, 'Cover Trees for Nearest Neighbor', ICML, pp. 97-104, DOI 10.1145/1143844.1143857) give EXACT NN queries in O(c^12 log n) where c is the Karger-Ruhl expansion constant — cost depends on intrinsic dimension (c ~ 2^d for intrinsic d), not ambient D. Real embeddings with low intrinsic dimension stay searchable at ambient D in the thousands. (2) Beyer et al. (1999) themselves identify workloads where concentration does NOT occur (e.g., well-separated clusters, query matching an exact point); Durrant & Kaban (2009, 'When is nearest neighbour meaningful: A converse theorem and implications', Journal of Complexity 25(4)) prove Euclidean distance does not concentrate for linear latent variable models when the number of relevant dimensions grows with ambient dimensions. (3) Mitigations, not reversals: Aggarwal, Hinneburg & Keim (2001, 'On the Surprising Behavior of Distance Metrics in High Dimensional Space', ICDT, pp. 420-434) show relative contrast decays slower for L1 and fractional Lp norms than L2 — a slower decay, still a decay. (4) Hubness (Radovanovic, Nanopoulos & Ivanovic 2010, 'Hubs in Space', JMLR 11:2487-2531) is an intrinsic high-D phenomenon — k-occurrence distributions become skewed — but it is generally a pathology for kNN semantics, not a pruning advantage. No source supports pruning getting sharper with raw dimensionality.

Sources:
- https://faculty.cc.gatech.edu/~isbell/reading/papers/cover-tree-icml.pdf
- https://dl.acm.org/doi/abs/10.1016/j.jco.2009.02.011
- https://dl.acm.org/doi/10.5555/645504.656414
- https://jmlr.org/papers/v11/radovanovic10a.html

Key facts:
- Distance concentration (Beyer et al., ICDT 1999): under broad conditions D_max/D_min -> 1 as dimension grows; the effect is empirically visible at as few as 10-15 dimensions.
- Weber, Schek & Blott (VLDB 1998): every partitioning/clustering index degrades to sequential scan in high D; sequential scan wins above ~10 dimensions; VA-file (compressed scan, no geometric pruning) beats hierarchical indexes above D ~ 5.
- KD-tree practical ceiling: roughly D < 20 (scikit-learn docs, O(D log N) below, nearly O(DN) and slower-than-brute-force above); ball trees extend this only when data is highly structured (low intrinsic dimension).
- Chavez et al. (ACM Computing Surveys 2001): metric-space pruning power is a function of the distance histogram's intrinsic dimensionality rho = mu^2/(2 sigma^2); high D means large mean + small variance = weak pruning; no exact technique handles intrinsic dimension beyond ~20.
- Angles concentrate at 90 degrees as D grows (Cai, Fan & Jiang, JMLR 2013); random inner products are O(1/sqrt(d)); hypercube diagonal-axis angle cosine is 1/sqrt(d) -> orthogonal. 'Sharper diagonal angles' is backwards.
- Exact NN beating brute force requires time or space exponential in dimension (Indyk & Motwani, STOC 1998) — the stated motivation for inventing LSH; HNSW (Malkov & Yashunin, TPAMI 42(4):824-836) is approximate and has linear-time adversarial instances (NeurIPS 2023).
- The only genuine escape hatch is LOW INTRINSIC dimension inside high ambient dimension: cover trees (Beygelzimer et al., ICML 2006) answer exact queries in O(c^12 log n) depending on the expansion constant, not ambient D. Benchmark implication: any observed pruning success on real embeddings measures intrinsic, not ambient, dimension.
- Triangle-inequality lower bounds in high D are directly documented as too loose to prune even with optimized landmarks (TRIM, arXiv:2508.17828, 2025).
- POC design implication: test the hypothesis on i.i.d. or whitened data at D = 2, 10, 20, 50, 100+; the hypothesis predicts pruning fraction rising with D, the literature predicts it collapsing toward 0% past D ~ 10-20. Include brute force and a low-intrinsic-dimension dataset (e.g., data on a low-D manifold embedded in high D) as the control where pruning legitimately survives.

## Prior art: hypercube memories and energy-based retrieval

The 'thermodynamic ripple search engine' concept is thoroughly anticipated in the literature; no component is novel. (1) Content-addressable memory on hypercube corners with activation within a Hamming radius IS Kanerva's Sparse Distributed Memory (1988): canonical design uses 1000-bit addresses, 1M hard locations, radius-451 activation, majority-vote readout, and a 'critical distance' inside which iterated reads converge. Its measured limits: capacity scales with hard-location count but information per connection equals Hopfield's (Keeler 1988), and it degrades on correlated real data because the uniform-random-address assumption fails. (2) Energy-function associative retrieval is Hopfield (1982), capacity 0.138N (Amit et al. 1985) / n/(2-4 log n) (McEliece et al. 1987); the thermodynamic variant is the Boltzmann machine (Ackley-Hinton-Sejnowski 1985); modern Hopfield networks (Ramsauer et al. 2020) store exponentially many continuous patterns, retrieve in one step, and are literally transformer attention. (3) Physical activation-propagation search exists in silicon: Frady et al. 2020 ran approximate k-NN on 768 Loihi chips using spike-latency codes (first k neurons to fire = k-NN), claiming better latency/index-build/energy than CPU baselines on >1M-pattern datasets. (4) The crux: Kanerva 2009 shows high dimensionality is a feature ONLY for designed random/symbolic codes in Hamming space, where concentration of measure (normalized distance 0.5 +/- 1/(2*sqrt(D))) makes vectors quasi-orthogonal and noise margins enormous; for metric nearest-neighbor on real correlated data, the same concentration destroys distance contrast (Beyer et al. 1999; Aggarwal et al. 2001; hubness, Radovanovic et al. 2010), so a ripple engine benchmarked on real embeddings sits in the unfavorable regime. POC implication: benchmark against SDM, modern-Hopfield/attention retrieval, and standard ANN baselines (HNSW/IVF); the hypothesis can only claim novelty via empirical wins, not mechanism.

### SUPPORTED: Content-addressable memory whose addresses are hypercube corners, with retrieval by activating all locations within a Hamming radius of the cue, already exists in the literature

This is exactly Kanerva's Sparse Distributed Memory (Kanerva, P., 1988, Sparse Distributed Memory, MIT Press; companion report: Flynn, Kanerva & Bhadkamkar, 1989, 'Sparse Distributed Memory: Principles and Operation', Stanford CSL-TR-89-400). Addresses are points of {0,1}^n (hypercube corners); a sparse random sample of 'hard locations' is physically instantiated; a read/write cue activates every hard location within Hamming radius r of the cue. Kanerva's canonical worked example: n=1000 (address space 2^1000), 1,000,000 hard locations, access radius r=451 bits chosen so ~1/1000 of hard locations activate; writes increment counters at all activated locations, reads pool activated counters and take a per-bit majority vote. Kanerva defines a 'critical distance': cues closer than it converge to the stored word under iterated reading, cues beyond it diverge. The 'ripple engine' hypothesis's core mechanism (hypercube-corner CAM + radius-limited activation) is therefore 38-year-old prior art, not a novel idea.

Sources:
- http://i.stanford.edu/pub/cstr/reports/csl/tr/89/400/CSL-TR-89-400.pdf
- https://papers.neurips.cc/paper_files/paper/2021/file/8171ac2c5544a5cb54ac0f38bf477af4-Paper.pdf
- https://www.frontiersin.org/articles/10.3389/fnhum.2014.00222/full

### SUPPORTED: SDM's measured strengths and limits are documented: noise-tolerant best-match retrieval and capacity that scales with hard-location count, but degraded performance on correlated (non-random) real data and a bounded critical distance

Strengths: iterated reads converge to a clean stored pattern when the cue is within the critical distance; one-shot incremental writes; graceful degradation (majority-vote pooling over ~1000 activated locations). Limits: (1) Keeler, J.D., 1988, 'Comparison Between Kanerva's SDM and Hopfield-type Neural Networks', Cognitive Science 12(3):299-329, showed SDM capacity grows with the number of hard locations independently of word dimension, but the information stored PER CONNECTION is the same constant as for Hopfield-type nets - SDM buys capacity with more hardware, not more efficiency per synapse. (2) Critical distance is bounded: in a sparsely populated memory it is necessarily less than the sum of storage and retrieval Hamming radii. (3) The uniform-random-address assumption breaks on real data: a NASA empirical study of SDM on discrete speech (NTRS 19920003219) notes clustered real inputs land far from uniformly scattered hard locations (it would take ~10^27 uniform locations for an arbitrary input to fall within 40 bits of one), motivating Keeler's fix of drawing hard-location addresses from the data distribution itself. Exact closed-form capacity constants (e.g., Chou 1988's exhaustive analysis) were not recoverable from the searched excerpts and should be read from the originals before quoting numbers.

Sources:
- https://philpapers.org/rec/KEECBK
- https://ntrs.nasa.gov/api/citations/19920003219/downloads/19920003219.pdf
- https://ntrs.nasa.gov/citations/19890041660
- http://i.stanford.edu/pub/cstr/reports/csl/tr/89/400/CSL-TR-89-400.pdf

### SUPPORTED: Associative memory driven by an energy function (states 'roll downhill' to stored attractors) exists, with precisely quantified capacity limits

Classical Hopfield network (Hopfield, J.J., 1982, PNAS 79:2554-2558) retrieves by descending an energy function to a stored attractor. Capacity is rigorously characterized: Amit, Gutfreund & Sompolinsky (1985, 'Storing infinite numbers of patterns in a spin-glass model of neural networks', Phys. Rev. Lett. 55:1530) found a first-order phase transition at ~0.138N patterns per N neurons (replica-symmetric; ~0.144N with one-step RSB per later refinements); McEliece, Posner, Rodemich & Venkatesh (1987, 'The Capacity of the Hopfield Associative Memory', IEEE Trans. Information Theory 33(4):461-482, DOI 10.1109/TIT.1987.1057328) proved asymptotic capacity n/(2 log n) for exact recovery of most random memories and n/(4 log n) for all of them. The stochastic/'thermodynamic' variant also exists: Ackley, Hinton & Sejnowski (1985, 'A Learning Algorithm for Boltzmann Machines', Cognitive Science 9(1):147-169) replaced deterministic Hopfield units with probabilistic ones so the state distribution is a Boltzmann (thermal) distribution over the energy landscape - i.e., 'thermodynamic' energy-based retrieval is 40-year-old prior art.

Sources:
- https://www.researchgate.net/publication/283617646_Storing_Infinite_Numbers_of_Patterns_in_a_Spin-Glass_Model_of_Neural_Networks
- https://www.semanticscholar.org/paper/The-capacity-of-the-Hopfield-associative-memory-McEliece-Posner/d5db1cc4a5ea1fcf017ceac7f07496292f37a9a2
- https://papers.cnl.salk.edu/PDFs/A%20learning%20algorithm%20for%20Boltzmann%20machines%201985-4616.pdf

### SUPPORTED: Modern Hopfield networks (Ramsauer et al. 2020) extend energy-based associative memory to continuous states with exponential capacity and one-step retrieval, and are equivalent to transformer attention

Ramsauer, H., Schaefl, B., Lehner, J., et al., 2020, 'Hopfield Networks is All You Need', arXiv:2008.02217 (published ICLR 2021): generalizes the exponential-interaction energy of Demircigil et al. (2017) to continuous states via E = -lse(beta, X^T xi) + 1/2 xi^T xi + beta^-1 ln N + 1/2 M^2; the update rule provably converges globally to stationary points, retrieves in one update with exponentially small error, and stores exponentially many (in dimension d) patterns - a qualitative jump past the 0.138N classical limit. The update rule IS transformer attention. Relevant to the hypothesis: 'energy-based content retrieval' is not only prior art, its modern form is the mechanism inside every transformer. Related: Bricken & Pehlevan (2021, 'Attention Approximates Sparse Distributed Memory', NeurIPS 2021) showed attention also approximates SDM's Hamming-sphere intersection, tying the SDM and Hopfield strands together.

Sources:
- https://arxiv.org/pdf/2008.02217
- https://ml-jku.github.io/hopfield-layers/
- https://papers.neurips.cc/paper_files/paper/2021/file/8171ac2c5544a5cb54ac0f38bf477af4-Paper.pdf

### SUPPORTED: Spiking/neuromorphic hardware performing nearest-neighbor search via propagating activation has published benchmark results (Intel Loihi)

Frady, E.P., Orchard, G., Florey, D., et al., 2020, 'Neuromorphic Nearest-Neighbor Search Using Intel's Pohoiki Springs', arXiv:2004.12691 (NICE 2020 workshop): approximate k-NN on 768 interconnected Loihi chips (~100M spiking neurons). Query encoded as spike-time latency code (earlier spike = larger magnitude); each integrate-and-fire neuron's membrane voltage represents one stored point's match; the first k neurons to fire are the k nearest neighbors. Abstract claims superior latency, index build time, and energy vs state-of-the-art CPU implementations on standard datasets of >1M high-dimensional patterns, plus O(1) online insertion. Secondary sources report ~3 ms query latency, ~6.6-7.3 mJ/query, and 30 s index build for 1M points vs hours for HNSW - these specific numbers come from summaries/excerpts, not the verified full text (proxy blocked the PDF), so re-verify against arXiv:2004.12691 before citing in the POC. Hardware CAM retrieval also exists outside spiking systems: Karunaratne et al., 2020, 'In-memory hyperdimensional computing', Nature Electronics 3:327-337 (DOI 10.1038/s41928-020-0410-3) - associative memory search over 760k phase-change memory devices with ~6x energy savings vs optimized CMOS.

Sources:
- https://arxiv.org/abs/2004.12691
- https://www.researchgate.net/publication/340963496_Neuromorphic_Nearest-Neighbor_Search_Using_Intel's_Pohoiki_Springs
- https://www.semanticscholar.org/paper/In-memory-hyperdimensional-computing-Karunaratne-Gallo/9a629da36abb17acf095ea5c1bc96200636fcfd5

### SUPPORTED: In hyperdimensional computing, high dimensionality is a feature: concentration of measure makes random vectors quasi-orthogonal (Kanerva 2009)

Kanerva, P., 2009, 'Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors', Cognitive Computation 1:139-159 (DOI 10.1007/s12559-009-9009-8). Distance between two random D-bit vectors is binomial: mean D/2, normalized std dev 1/(2*sqrt(D)) ~= 0.005 at D=10,000, so nearly all random pairs sit at normalized distance ~0.47-0.53 - i.e., quasi-orthogonal. Consequences exploited by design: a fresh random vector is almost surely ~5000 bits from everything already stored (huge noise margin); a pattern corrupted in even a third of its bits remains far closer to its original than to any unrelated vector; bundling/binding algebra works because cross-talk is predictably near-orthogonal. This is the regime where the ripple-engine intuition is CORRECT - but only for high-dimensional RANDOM/SYMBOLIC codes in Hamming space, which is precisely Kanerva's published framework (SDM 1988 + HD computing 2009), not a new discovery.

Sources:
- https://link.springer.com/article/10.1007/s12559-009-9009-8
- https://redwood.berkeley.edu/wp-content/uploads/2018/01/kanerva2009hyperdimensional.pdf
- https://arxiv.org/pdf/2111.06077

### SUPPORTED: For metric nearest-neighbor search on real-world (correlated, real-valued) data, high dimensionality hurts: distance contrast collapses and NN can become meaningless

Beyer, K., Goldstein, J., Ramakrishnan, R., Shaft, U., 1999, 'When Is Nearest Neighbor Meaningful?', ICDT 1999, LNCS 1540, pp. 217-235: under broad conditions (broader than i.i.d. dimensions), distance to the nearest point approaches distance to the farthest point as dimensionality grows; empirically the effect bites at as few as 10-15 dimensions. Crucial caveat in the same paper: NN is NOT always meaningless in high-D - it stays meaningful for workloads with structure (e.g., well-separated clusters, low intrinsic dimensionality). Aggarwal, Hinneburg & Keim, 2001, 'On the Surprising Behavior of Distance Metrics in High Dimensional Space', ICDT 2001, LNCS 1973, pp. 420-434: meaningfulness depends on the norm; smaller k in L_k (and fractional norms) preserves contrast better. Radovanovic, Nanopoulos & Ivanovic, 2010, 'Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data', JMLR 11:2487-2531: in high-D real data a few 'hub' points become nearest neighbors of disproportionately many queries, further distorting retrieval.

Sources:
- https://dblp.org/rec/conf/icdt/BeyerGRS99.html
- https://members.loria.fr/MOBerger/Enseignement/Master2/Exposes/beyer.pdf
- https://www.semanticscholar.org/paper/On-the-Surprising-Behavior-of-Distance-Metrics-in-Aggarwal-Hinneburg/88fc7fc6ed73db6251967cfe40507a92008e5949
- https://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf

### REFUTED: Retrieval by activation 'rippling'/propagating outward from a cue through an associative structure is a novel mechanism

Spreading activation is a 50-year-old retrieval model: Collins, A.M. & Loftus, E.F., 1975, 'A Spreading-Activation Theory of Semantic Processing', Psychological Review 82(6):407-428 - activation spreads from a cued concept node along network edges with gradual falloff with distance, and the framework was subsequently applied to information retrieval (nodes as documents/terms). Combined with SDM's radius-limited activation (1988), Hopfield/Boltzmann energy relaxation (1982/1985), and spike-wavefront k-NN on Loihi (2020), every mechanistic ingredient of the 'thermodynamic ripple search engine' - hypercube-corner CAM, Hamming-radius activation, energy-function dynamics, thermal noise, physical propagation implementing search - has published prior art. Novelty, if any, could lie only in a specific new combination or in beating the published baselines empirically, which is what the POC benchmark should test.

Sources:
- https://en.wikipedia.org/wiki/Spreading_activation
- https://www.bibsonomy.org/bibtex/e55141080b2fcdfb31e5c4a4ce53d6bd
- http://i.stanford.edu/pub/cstr/reports/csl/tr/89/400/CSL-TR-89-400.pdf
- https://arxiv.org/abs/2004.12691

Key facts:
- Kanerva SDM (1988): addresses are corners of {0,1}^n; canonical config n=1000, 2^1000 address space, 1,000,000 hard locations, access radius 451 bits activating ~1/1000 of locations; read = per-bit majority vote over activated counters; 'critical distance' separates converging from diverging cues; average distance from a random point to its nearest hard location ~424 bits.
- Keeler (1988, Cognitive Science 12:299-329): SDM capacity scales with number of hard locations independent of word dimension, but information per connection is the same constant as Hopfield - no free lunch per synapse.
- SDM assumes uniformly random addresses; on clustered real data the uniform hard-location assumption fails badly (NASA NTRS 19920003219: ~10^27 uniform locations needed for any input to fall within 40 bits of one), fixed by data-distributed hard locations.
- Classical Hopfield capacity: ~0.138N patterns (Amit-Gutfreund-Sompolinsky 1985, phase transition), n/(2 log n) for most / n/(4 log n) for all patterns exactly recoverable (McEliece et al. 1987, IEEE Trans. IT 33:461-482).
- Modern Hopfield (Ramsauer et al. 2020, arXiv:2008.02217): continuous states, log-sum-exp energy, one-update retrieval with exponentially small error, exponential-in-d capacity; update rule = transformer attention; Bricken & Pehlevan (NeurIPS 2021) link attention to SDM as well.
- Boltzmann machine (Ackley, Hinton & Sejnowski 1985, Cognitive Science 9:147-169) = stochastic Hopfield with temperature; the 'thermodynamic' element of the hypothesis is 1985 prior art.
- Frady et al. 2020 (arXiv:2004.12691): approximate k-NN on 768-chip Loihi system (100M neurons), spike-latency coding, first-k-neurons-to-fire = k nearest neighbors; claims superior latency/index-build/energy vs CPU on >1M-pattern datasets and O(1) insertion. Secondary (unverified vs full text): ~3 ms/query, ~6.6-7.3 mJ/query, 1M-point index built in ~30 s.
- Kanerva 2009 (Cognitive Computation 1:139-159): random D-bit vectors concentrate at normalized Hamming distance 0.5 with std dev 1/(2*sqrt(D)) (~0.005 at D=10,000), making random vectors quasi-orthogonal - the mathematical reason high-D is a FEATURE for random/symbolic codes.
- Beyer et al. 1999 (ICDT): on broad data models, nearest and farthest distances converge as d grows (effects from 10-15 dims), so metric NN on real high-D data loses meaning - UNLESS intrinsic dimension is low or clusters are separated; Aggarwal et al. 2001: effect worsens with larger L_k exponents; Radovanovic et al. 2010 (JMLR 11): hubness further skews high-D NN.
- The crux resolved: high-D helps when the CODE IS DESIGNED (i.i.d. random/symbolic hypervectors, Hamming space, task = recover a stored item from a noisy copy - interference is predictably quasi-orthogonal). High-D hurts when VECTORS ARE GIVEN real data (correlated, low intrinsic dimension embedded in high ambient dimension, task = rank semantic proximity - distance contrast collapses). A ripple-engine POC on real embeddings operates in the second regime, where the HD-computing advantages do not transfer.
- Tooling note: the sandbox proxy policy-denied direct fetches of arxiv.org/rctn.org/stanford PDFs (CONNECT 403), so numeric details flagged as secondary above were cross-checked only via search-indexed excerpts, not full texts.

## Prior art: self-strengthening search paths

The hypothesis's claimed novelty is refuted by the literature. 'Searches leave pheromone-like traces that strengthen paths' is Ant Colony Optimization routing (AntNet, Di Caro & Dorigo, JAIR 9:317-365, 1998 — superior to OSPF/Bellman-Ford/Q-routing in simulation, but with the canonical stagnation failure mode: reinforcement without evaporation collapses all traffic onto one sub-optimal path). 'Frequently accessed nodes get shorter paths' is self-adjusting data structures (splay trees, Sleator & Tarjan, JACM 1985, static optimality theorem, with write-on-read and adversarial-sequence costs). 'The index reorganizes itself from the query workload' is database cracking / adaptive indexing (Idreos et al., CIDR 2007, Test-of-Time Award), whose independently measured failure modes — sequential-pattern worst cases, high variance under skew, slow convergence — were fixed by adding random (non-workload-driven) reorganization (Stochastic Cracking, PVLDB 2012). 'Activation ripples outward through an association graph' is spreading activation (Crestani, AI Review 1997): unconstrained spreading provably degenerates to query-independent results, so decay < 1 and hop/fan-out limits are mandatory. Learned indexes (Kraska et al., SIGMOD 2018) show the same pattern at the structure level: big read-only wins (up to 70% vs B-trees per the authors; independently confirmed in SOSD, PVLDB 14(1)) traded against unsolved updates, distribution shift, and adversarial fragility. HNSW itself (Malkov & Yashunin, TPAMI 2020) is query-static — built by insertion, never rewired by queries — and the gap the hypothesis gestures at is already an active subfield: RoarGraph (VLDB 2024) rebuilds ANN graphs from historical queries after measuring that OOD queries sit 2.1-11.3x farther from their neighbors, and GATE (2025) learns query-aware entry points for 1.2-2.0x QPS over HNSW. Across all six lines, the recurring empirical result is the same trade: adapting a structure to its workload buys average-case speed on that workload and costs robustness on out-of-distribution, sequential, or shifted workloads — and permanent (non-decaying) reinforcement is the documented failure configuration in both ACO and spreading activation. The POC should therefore benchmark against static HNSW plus a query-aware baseline (RoarGraph/GATE-style), sweep decay/evaporation (including the hypothesis's implied decay=0), split ID vs OOD recall, use sequential and shifting workloads as stress cases, and report variance and write-amplification, not just mean latency.

### REFUTED: The hypothesis's core mechanism — 'every search leaves a residual trace like an ant pheromone trail; highly traveled paths burn into permanent shortcuts; frequently co-activated nodes get stronger/shorter connections' — is novel.

Every component has decades of prior art. (1) Pheromone-trace search reinforcement on graphs is literally Ant Colony Optimization: Di Caro & Dorigo, 'AntNet: Distributed Stigmergetic Control for Communications Networks', JAIR vol. 9, pp. 317-365 (1998) maintains per-node pheromone routing tables that are reinforced by traversals and decay by evaporation. (2) 'Frequently accessed items get shorter access paths' is the defining property of self-adjusting data structures: Sleator & Tarjan, 'Self-Adjusting Binary Search Trees', JACM 32(3):652-686 (1985) — splay trees move accessed nodes to the root on every access and are provably within a constant factor of the static optimum tree for any access distribution. (3) Indexes that self-organize as a side effect of queries is database cracking: Idreos, Kersten & Manegold, 'Database Cracking', CIDR 2007, pp. 68-78 (winner of the CIDR Test-of-Time Award for opening the adaptive-indexing research direction). (4) Activation flowing through an association graph at query time is spreading activation: Crestani, 'Application of Spreading Activation Techniques in Information Retrieval', Artificial Intelligence Review 11:453-482 (1997), surveying work going back to the 1960s-80s. (5) Restructuring a proximity graph from historical queries exists: RoarGraph (Chen et al., PVLDB/VLDB 2024, arXiv:2408.08933) builds a projected bipartite graph from past queries. What is genuinely less explored: continuous per-query Hebbian edge-weight updates inside a production ANN vector index (HNSW-style) — but that is a gap in one niche, not a novel mechanism.

Sources:
- https://arxiv.org/abs/1105.5449
- https://dl.acm.org/doi/10.1145/3828.3835
- https://stratos.seas.harvard.edu/publications/database-cracking
- https://link.springer.com/article/10.1023/A:1006569829653
- https://arxiv.org/pdf/2408.08933

### SUPPORTED: HNSW (Malkov & Yashunin) builds its graph by incremental insertion and does NOT adapt edges to the query workload — queries are read-only traversals.

Malkov & Yashunin, 'Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs', IEEE TPAMI 42(4):824-836 (2020; arXiv:1603.09320, 2016). Abstract: the method 'incrementally builds a multi-layer structure consisting from hierarchical set of proximity graphs (layers) for nested subsets of the stored elements'; layer assignment is random with exponentially decaying probability; search descends fixed layers greedily. No mechanism in the paper modifies edges in response to queries — structure changes only on element insertion. Measured benefit (authors' claim): 'strongly outperforms previous open-source state-of-the-art vector-only approaches', with logarithmic complexity scaling attributed to the layer hierarchy. Known weakness relevant to the POC: entry-point/edge choices are query-oblivious, which is exactly what GATE (arXiv:2506.15986, reporting 1.2-2.0x QPS over HNSW/NSG at equal recall via learned query-aware entry points) and RoarGraph exploit. Also note arXiv:2412.01940 ('the H in HNSW stands for Hubs') disputes that the hierarchy itself is necessary at scale.

Sources:
- https://arxiv.org/abs/1603.09320
- https://dl.acm.org/doi/10.1109/TPAMI.2018.2889473
- https://arxiv.org/abs/2506.15986
- https://arxiv.org/pdf/2412.01940

### SUPPORTED: Database cracking / adaptive indexing (Idreos et al., CIDR 2007+) is established prior art for indexes that self-organize from the query workload, and it has measured, well-characterized failure modes under adversarial/shifted workloads.

Idreos, Kersten & Manegold, 'Database Cracking', CIDR 2007, pp. 68-78: each range query physically partitions ('cracks') the column around its predicates, so the index emerges incrementally as a side effect of query processing; later queries over touched ranges get faster. Follow-ups: 'Updating a Cracked Database' (SIGMOD 2007), self-organizing tuple reconstruction (SIGMOD 2009). Failure modes are measured in two primary sources: Halim, Idreos, Karras & Yap, 'Stochastic Database Cracking: Towards Robust Adaptive Indexing in Main-Memory Column-Stores', PVLDB 5(6):502-513 (2012) shows sequential query patterns are a near-worst case (each crack touches a large tuple count that falls only negligibly), and fixes it by injecting RANDOM reorganization steps — i.e., pure workload-following is fragile and needs stochastic noise. Schuhknecht, Jindal & Dittrich, 'The Uncracked Pieces in Database Cracking', PVLDB 7(2):97-108 (VLDB 2014 Best Paper; extended as VLDB Journal 2016, DOI 10.1007/s00778-015-0397-y), the first independent evaluation, confirms: slow/unreliable convergence to full-index performance is 'a major criticism', 'sequential workloads are the weak spots of query driven methods while skewed patterns increase the overall variance'; convergence is fastest around ~10% selectivity.

Sources:
- https://stratos.seas.harvard.edu/publications/database-cracking
- http://vldb.org/pvldb/vol5/p502_felixhalim_vldb2012.pdf
- http://www.vldb.org/pvldb/vol7/p97-schuhknecht.pdf
- https://link.springer.com/article/10.1007/s00778-015-0397-y
- https://www.cwi.nl/en/news/tidying-up-while-you-search-cwi-paper-database-cracking-wins-cidr-test-of-time-award/

### SUPPORTED: Spreading activation in IR (Crestani 1997) is established prior art for 'activation rippling through an association graph', and unconstrained spreading is known to fail.

Crestani, 'Application of Spreading Activation Techniques in Information Retrieval', Artificial Intelligence Review 11:453-482 (Dec 1997), DOI 10.1023/A:1006569829653: surveys SA over semantic/association networks for associative IR and concludes SA is useful only as a complementary module, constrained to a few hops from query nodes with a decay factor < 1. Known failure modes, confirmed in the literature: pure/unconstrained SA converges to a fixed state where results become nearly independent of the input query (i.e., 'pure spreading activation is pointless'); heuristic constraints (hop limits, fan-out limits, thresholds, decay) are essential, not optional; traversal becomes 'query-blind' once seeded; high false-positive rates on non-relevant entities; parameters are notoriously hard to tune; 1980s evaluations found standalone SA does not reach high retrieval performance. This failure mode maps directly onto the 'ripple' hypothesis: without decay/evaporation and fan-out control, residual traces wash out query specificity.

Sources:
- https://link.springer.com/article/10.1023/A:1006569829653
- https://arxiv.org/pdf/2512.15922
- https://www.researchgate.net/publication/234812467_On_the_Use_of_Spreading_Activation_Methods_in_Automatic_Information_Retrieval

### SUPPORTED: Ant colony optimization for routing (AntNet) is established prior art for pheromone-trail path reinforcement, with measured benefit and the stagnation failure mode.

Di Caro & Dorigo, 'AntNet: Distributed Stigmergetic Control for Communications Networks', JAIR 9:317-365 (1998): mobile agents ('ants') sample paths and deposit reinforcement in per-node probabilistic routing (pheromone) tables; good paths get stronger, pheromone evaporates over time. Measured benefit (authors' claim): compared against six state-of-the-art routing algorithms (the comparison set in the companion paper names OSPF, SPF, distributed adaptive Bellman-Ford, Q-routing, Predictive Q-routing) on realistic simulated IP networks, AntNet 'showed superior performance under all the experimental conditions' on throughput and packet delay — note this is simulation-only and was never adopted in production Internet routing. Canonical failure mode (Dorigo & Stützle, 'Ant Colony Optimization', MIT Press 2004; Dorigo & Di Caro survey chapters): STAGNATION — positive feedback alone makes all ants converge on one, generally sub-optimal, path ('with only pheromone amplification at work, this leads to rapid emergence of a stagnation situation... tours which, in general, are strongly sub-optimal'). Mandatory countermeasures: pheromone evaporation, pheromone bounds (MAX-MIN Ant System, Stützle & Hoos), and trail smoothing. Direct lesson for the hypothesis: 'permanent shortcuts' with no evaporation is the textbook-documented failure configuration.

Sources:
- https://arxiv.org/abs/1105.5449
- https://mlanthology.org/jair/1998/caro1998jair-antnet/
- https://link.springer.com/chapter/10.1007/BFb0056909
- http://www.scholarpedia.org/article/Ant_colony_optimization
- https://web2.qatar.cmu.edu/~gdicaro/15382/additional/aco-book.pdf

### SUPPORTED: Learned indexes (Kraska et al. 2018) deliver measured speedups on read-only workloads but degrade under updates and distribution shift.

Kraska, Beutel, Chi, Dean & Polyzotis, 'The Case for Learned Index Structures', SIGMOD 2018 (arXiv:1712.01208): learned (RMI) indexes 'outperform cache-optimized B-Trees by up to 70% in speed while saving an order-of-magnitude in memory' — authors' own numbers, read-only analytical workloads only; the paper itself lists write-heavy workloads and distribution change as open problems with no B-Tree-like guarantees. Independent benchmark: Marcus et al., 'Benchmarking Learned Indexes' (SOSD), PVLDB 14(1):1-13 (2020): RMI/RadixSpline do beat ART/FAST/B-trees on in-memory read-only lookups (e.g., RMI ~181ns vs B-tree ~482ns in one reported configuration) but RMIs are slow to build. Caveats from follow-up literature: lookup latency averages 2.92x higher on real-world datasets than same-sized synthetic ones; some studies find learned indexes equal or worse than algorithmic indexes on real-world distributions; robustness studies show models 'struggle to generalize beyond their training distributions'; adversarial insertions can crash the updatable ALEX index (out-of-memory in seconds, per 'Learned-Database Systems Security', arXiv:2212.10318). Net: workload-fitted structures trade worst-case robustness for average-case speed — the exact trade the ripple hypothesis must be benchmarked on.

Sources:
- https://arxiv.org/pdf/1712.01208
- https://vldb.org/pvldb/vol14/p1-marcus.pdf
- https://www.vldb.org/pvldb/vol15/p1079-maltry.pdf
- https://arxiv.org/pdf/2212.10318
- http://databasearchitects.blogspot.com/2017/12/the-case-for-b-tree-index-structures.html

### SUPPORTED: Query-adaptive proximity graphs already exist, and the literature quantifies how workload adaptation interacts with out-of-distribution queries.

Three primary-source lines of work: (1) RoarGraph (arXiv:2408.08933, VLDB 2024) rebuilds the ANN graph structure itself from historical queries via a projected bipartite graph, motivated by measurements that OOD (cross-modal) queries land 2.1x / 5.3x / 11.3x farther from their true nearest neighbors than in-distribution queries on tested datasets — i.e., query-distribution mismatch is measured to badly hurt standard proximity graphs. (2) GATE (arXiv:2506.15986, 2025) learns query-aware entry points over hub nodes with a two-tower contrastive model trained on historical queries: 1.2-2.0x QPS over HNSW/NSG at equal recall, with a reported ~1.2% in-distribution vs OOD gap (secondary summary, verify in paper). (3) OOD-DiskANN (arXiv:2211.12850) targets the same OOD problem for disk-based ANN. None of these do continuous per-query edge reinforcement at serve time; all adapt offline/periodically from logged queries. The direction of the evidence: adapting a graph to a workload helps that workload, and the OOD penalty is real, measured, and the central evaluation axis any 'ripple' POC must include.

Sources:
- https://arxiv.org/pdf/2408.08933
- https://arxiv.org/abs/2506.15986
- https://arxiv.org/pdf/2510.05975

### SUPPORTED: A search structure where frequently accessed items get provably shorter access paths, with known costs of that adaptation, predates all of the above (classical data structures).

Sleator & Tarjan, 'Self-Adjusting Binary Search Trees', JACM 32(3):652-686 (July 1985): splay trees restructure on EVERY access (accessed node splayed to root), giving O(log n) amortized operations and the static optimality theorem — for sufficiently long sequences, total access time is within a constant factor of the optimal fixed tree for that access distribution, i.e., frequently accessed items provably end up near the root. Known costs of self-adjustment, from the same literature: every read performs writes (bad for caches and concurrency), single operations can cost O(n) (only amortized bounds hold), uniform-random workloads make the restructuring pure overhead versus a static balanced tree, and the dynamic optimality conjecture (splay trees competitive with any offline rotation-based tree) remains unproven after 40 years. This is the cleanest theoretical frame for the hypothesis: 'paths burn into shortcuts' is self-adjustment, and its costs (write-on-read, adversarial sequences, no worst-case per-op guarantee) are textbook material.

Sources:
- https://dl.acm.org/doi/10.1145/3828.3835
- https://en.wikipedia.org/wiki/Splay_tree
- https://arxiv.org/pdf/1306.0207

Key facts:
- Novelty verdict: the 'thermodynamic ripple / pheromone trace / self-strengthening path' mechanism is a recombination of at least five established research lines (ACO routing 1998, splay trees 1985, database cracking 2007, spreading activation 1960s-1997, query-adaptive ANN graphs 2022-2025); the only arguable gap is continuous online per-query edge reinforcement inside a vector ANN index, which existing work does offline/batch instead.
- HNSW (Malkov & Yashunin, TPAMI 42(4):824-836, 2020; arXiv:1603.09320) is the natural POC baseline and is query-STATIC: built by incremental insertion with random layer assignment; queries never modify the graph; claimed logarithmic search scaling and 'strongly outperforms previous open-source state-of-the-art vector-only approaches'.
- Database cracking (Idreos/Kersten/Manegold, CIDR 2007; CIDR Test-of-Time Award) = indexes as a side effect of queries; independently measured failure modes (Schuhknecht et al., PVLDB 7(2):97-108, VLDB 2014 Best Paper): slow convergence to full-index performance, sequential query patterns are the weak spot, skewed patterns inflate variance; the field's fix was to inject RANDOMNESS (Stochastic Cracking, Halim et al., PVLDB 5(6):502-513, 2012) — pure workload-following is fragile.
- ACO's documented failure mode is STAGNATION: pheromone amplification without evaporation makes all agents converge on one generally sub-optimal path (Dorigo & Stützle, ACO, MIT Press 2004; Scholarpedia). The hypothesis's 'highly traveled paths burn into PERMANENT shortcuts' with no decay is precisely the known-broken configuration; every working ACO variant requires evaporation, pheromone bounds (MAX-MIN AS), or smoothing.
- Spreading activation's documented failure mode: unconstrained spreading converges to a query-independent fixed point (results stop depending on the query); Crestani (AI Review 11:453-482, 1997) concluded SA works only constrained to few hops with decay factor < 1 — the 'ripple' needs built-in decay and fan-out limits or it destroys retrieval precision.
- Learned indexes (Kraska et al., SIGMOD 2018): authors' claim up to 70% faster than cache-optimized B-trees with ~10x less memory, READ-ONLY only; independent SOSD benchmark (Marcus et al., PVLDB 14(1), 2020) confirms read-only wins but notes slow builds, ~2.92x worse latency on real-world vs synthetic data in follow-ups, open problems on updates and distribution shift, and adversarial-insertion crashes on updatable variants (ALEX OOM, arXiv:2212.10318).
- OOD penalty of workload adaptation is quantified: RoarGraph (VLDB 2024) measured OOD queries landing 2.1x/5.3x/11.3x farther from true neighbors than in-distribution queries, and fixes it by rebuilding the graph FROM historical queries; GATE (arXiv:2506.15986) adapts entry points (1.2-2.0x QPS over HNSW/NSG at equal recall). So 'adapt graph to workload' is an active 2024-2025 subfield, not open territory.
- Closest classical prior art for 'frequently used paths get shorter': splay trees (Sleator & Tarjan, JACM 32(3):652-686, 1985) with the static optimality theorem; costs of self-adjustment are textbook — writes on every read, O(n) single-op worst case, overhead under uniform workloads, dynamic optimality still unproven.
- POC benchmark design implications from the literature: (a) must include a decay/evaporation parameter sweep (no-decay is the known-failure configuration in both ACO and SA); (b) must test sequential/shifting and uniform-random query workloads, the measured worst cases for adaptive methods; (c) must report in-distribution vs out-of-distribution recall separately (RoarGraph methodology); (d) baselines should be static HNSW plus a workload-aware one (RoarGraph or GATE-style entry points), and report variance across runs, not just means (cracking literature shows adaptation inflates variance); (e) concurrency/write-amplification cost of mutating the index on every read is a first-class metric (splay-tree lesson).

## Prior art: decay-based caching and neighborhood prefetching

The 'thermodynamic ripple search engine' caching hypothesis claims novelty for three components, and the literature refutes each. (1) Decay/temperature eviction: LRFU (2001) is an exponential-decay recency+frequency score with O(1) 'heat up on access, cool over time' updates; 'evict by decayed temperature' with a fixed heat quantum per access is the same recurrence as LRFU/EXD up to reparameterization (variable quanta reduce it to GreedyDual/GDSF; non-exponential decay loses O(1) updates and lands in LRU-K/ACT-R territory). Production systems already ship it: Redis LFU with lfu-decay-time, TinyLFU's periodic counter halving, Oracle's touch-count hot/cold regions, and Teradata's hot/warm/cold data-temperature tiering with automatic sink-to-disk migration. (2) 'Natural dissipation without bookkeeping': Hyperbolic Caching (ATC 2017) and Redis both evict via lazily-decayed priorities and random sampling with no ordered structures. (3) Neighborhood prefetch/keep-warm: semantic-region caching with distance-based replacement (Dar et al., VLDB 1996), semantic-distance cluster hoarding (SEER, SOSP 1997), Markov/correlation prefetchers fetching multiple predicted successors (ISCA 1997), and modern LLM semantic caches (GPTCache et al.) cover it; the conceptual mechanism (decay + spreading activation over a semantic graph) is Collins & Loftus 1975 / ACT-R. Measured baselines the POC must beat: ARC over LRU (4.24%->23.82% at 16MB workstation; 9.19%->20% SPC1-like at 4GB), LRU-2 over LRU (>2x hit ratio at small buffers), Markov prefetching (54% stall reduction). Equally important, prefetching is documented to HURT under low accuracy via cache pollution and bandwidth contention (Srinath HPCA 2007; Lee/Kim/Vuduc TACO 2012), so the benchmark must include low-semantic-locality workloads and report prefetch accuracy/pollution, not just hit rate. Residual white space is narrow: semantic-aware eviction for embedding caches is an acknowledged open problem (GPTCache still uses LRU/FIFO; offline-optimal semantic caching is NP-hard), and no system was found implementing literal coupled heat diffusion between cached items - but that formulation's intended effects are fully anticipated, so the POC should be framed as testing a recombination against strong baselines (ARC, W-TinyLFU, hyperbolic, GDSF, LRU-K, plus Markov/semantic prefetchers), not as validating a novel paradigm against plain LRU.

### REFUTED: Novelty claim: eviction by 'decayed temperature' (data cools over time, hot data stays cached) is new relative to LRU/LFU-family bookkeeping.

Decayed-score eviction is a 25+ year-old, repeatedly reinvented family. LRFU (Lee, Choi, Kim, Noh, Min, Cho, Kim, 'LRFU: A Spectrum of Policies that Subsumes the Least Recently Used and Least Frequently Used Policies', IEEE Transactions on Computers 50(12):1352-1361, Dec 2001, DOI 10.1109/TC.2001.970573) assigns every block a Combined Recency and Frequency (CRF) value in which each past reference contributes via a monotonically non-increasing weighing function F(x) (exponential form (1/p)^(lambda*x)); lambda tunes a continuous spectrum with LFU and LRU as endpoints. The EXD policy used in tiered storage (documented in 'Automating Distributed Tiered Storage Management in Cluster Computing', arXiv:1907.02394) is literally the temperature recurrence W = 1 + W*e^(-alpha*(t_now - t_last_access)). Redis's production LFU (official docs, redis.io 'Key eviction') is an 8-bit logarithmic counter incremented probabilistically (lfu-log-factor, default 10) and decayed by elapsed minutes (lfu-decay-time, default 1; 0 = never decay). TinyLFU (Einziger, Friedman, Manes, ACM Transactions on Storage 13(4), 2017; earlier PDP 2014 / arXiv:1512.00727) ages a count-min-style frequency sketch by periodically halving all counters. GDSF (Cherkasova, HP Labs HPL-98-69, 1998) combines an inflating aging clock + frequency + size. Oracle's buffer cache uses a touch-count algorithm with hot/cold LRU regions, midpoint insertion, time-gated counter increments (~3s via _db_aging_touch_time) and counter reset on promotion (third-party documentation only; hidden underscore parameters, not official Oracle docs). Teradata Virtual Storage (vendor white papers EB6690, EB6407, ~2008-2011) assigns per-cylinder 'data temperature' (hot/warm/cold) and automatically migrates data between SSD and HDD as temperature changes - the exact 'cools and sinks back to disk' tiering. Stoica, Levandoski, Larson ('Identifying Hot and Cold Data in Main-Memory Databases', ICDE 2013, pp. 26-37) classify record hotness with four exponential-smoothing estimators and report higher hit rates than the best caching techniques.

Sources:
- https://www.openu.ac.il/home/wiseman/2os/lru/lrfu.pdf
- https://arxiv.org/pdf/1907.02394
- https://redis.io/docs/latest/develop/reference/eviction/
- https://atsjp.github.io/JavaLearnMap/Java/Cache/Caffeine.assets/TinyLFU_PDP2014.pdf
- https://eclass.uoa.gr/modules/document/file.php/D245/2015/HPL-98-69R1_GDS.pdf
- https://www.adp-gmbh.ch/ora/concepts/cache.html
- https://jonathanlewis.wordpress.com/oracle-core/oc-5-caches-and-copies/
- https://assets.teradata.com/resourceCenter/downloads/WhitePapers/EB6690.pdf
- https://www.microsoft.com/en-us/research/wp-content/uploads/2013/04/ColdDataClassification-icde2013-cr.pdf

### REFUTED: The hypothesis's implied contrast: 'natural dissipation' eviction avoids the bookkeeping data structures (lists/heaps) that LRU/LFU require.

Hyperbolic Caching (Blankstein, Sen, Freedman, USENIX ATC 2017, pp. 499-511) was built on exactly this observation: it states that existing strategies 'rely on data structures to maintain a total ordering of the cached items', abandons ordered structures entirely, gives each item a lazily-evaluated decaying priority (accesses divided by time in cache - a hyperbolic, not exponential, decay), and evicts by sampling 64 random items and discarding the lowest-priority one. Redis implements LRU and LFU the same way in production: approximate, no global ordering, random sampling at eviction time, counters decayed lazily only when a key is sampled. So 'score decays on its own; evict the coldest sampled item; no list maintenance' is established engineering practice, not a new eviction paradigm.

Sources:
- https://collaborate.princeton.edu/en/publications/hyperbolic-caching-flexible-caching-for-web-applications/
- https://www.microsoft.com/en-us/research/publication/hyperbolic-caching-flexible-caching-for-web-applications/
- https://redis.io/docs/latest/develop/reference/eviction/

### REFUTED: 'Evict by decayed temperature' is mathematically different from LRFU's combined recency+frequency with exponential decay.

They are the same recurrence up to reparameterization. LRFU's CRF is C(t) = sum_i F(t - t_i) over past references with exponential weighing F(x) = (1/p)^(lambda*x); the multiplicative property of the exponential (F(x+y) = F(x)F(y)) is precisely what permits the O(1) incremental update C_new = 1 + F(delta)*C_old, i.e. 'add a fixed quantum of heat on access, multiply by a decay factor for elapsed time'. A temperature T updated as T <- T*e^(-alpha*dt) + Q per access is identical with alpha = lambda*ln(p) and score scaled by Q. The EXD policy's published formula W = 1 + W*e^(-alpha*(timeNow - timeLastAccess)) makes the identity explicit in the literature. The only ways a 'temperature' rule differs mathematically: (a) variable heat quantum per access (by size/cost) - which reduces it to the GreedyDual/GDSF class (Cherkasova 1998); (b) non-exponential decay (power-law etc.) - which breaks the O(1) incremental update and is the regime LRU-K (O'Neil, O'Neil, Weikum, SIGMOD 1993) and ACT-R-style power-law decay occupy; (c) cross-item heat transfer ('radiation' between neighbors) - no exact match found for coupled item-to-item decay dynamics in eviction scoring, but its intended effect (keep semantically related items warm) is covered by semantic-region caching and clustering-based hoarding. Caveat: the exact base of F(x) in Lee et al. 2001 was confirmed only via secondary sources as (1/p)^(lambda*x) (the primary PDF was unreachable through this session's proxy); a follow-up competitive-analysis paper uses 1/2 <= lambda <= 1. LRFU's own authors note performance 'depends crucially on the choice of lambda' - the same tuning burden any temperature system will face.

Sources:
- https://www.openu.ac.il/home/wiseman/2os/lru/lrfu.pdf
- https://arxiv.org/pdf/1907.02394
- https://blog.shunzi.tech/post/CRP-Coarse-Grained-Two-Frequency/
- https://researchgate.net/publication/2928223_Competitive_Analysis_of_the_LRFU_Paging_Algorithm
- https://eclass.uoa.gr/modules/document/file.php/D245/2015/HPL-98-69R1_GDS.pdf
- https://dl.acm.org/doi/10.1145/170035.170081

### REFUTED: Novelty claim: pulling a whole semantic/topological NEIGHBORHOOD into cache along with the requested item ('activated neighborhood heats up') is new.

At least four independent literatures do this. (1) Semantic caching in databases: Dar, Franklin, Jonsson, Srivastava, Tan, 'Semantic Data Caching and Replacement', VLDB 1996, pp. 330-341 - the cache is organized as dynamically defined 'semantic regions' (constraint formulas over tuples), queries split into a locally-answered probe and a 'remainder query'; replacement is ranked per-region using semantic locality rather than pure temporal locality, with a distance-based value computed from the 'center of gravity' of recent queries (center-of-gravity language confirmed in paper snippet; the commonly cited metric is Manhattan distance - full formula not independently re-verifiable this session because the proxy blocked the PDF). (2) File hoarding: Kuenning & Popek, 'Automated Hoarding for Mobile Computers', SOSP 1997 (SEER) - defines a 'semantic distance' between files, runs agglomerative clustering, and hoards entire related clusters, explicitly arguing this 'has the potential to perform much better than LRU-style caching algorithms'. (3) Hardware/memory prefetching of correlated neighborhoods: Joseph & Grunwald, 'Prefetching using Markov Predictors', ISCA 1997, pp. 252-263, DOI 10.1145/264107.264207 - a correlation table approximating a Markov graph over addresses prefetches MULTIPLE predicted successors per miss (reported 54% average reduction in memory-operation stalls on commercial benchmarks); stride/RPT prefetching (Chen & Baer, 'Effective Hardware-Based Data Prefetching for High-Performance Processors', IEEE Trans. Computers, 1995); sequential one-block-lookahead prefetch (Smith, 'Sequential Program Prefetching in Memory Hierarchies', IEEE Computer 11(12):7-21, 1978; survey in VanderWiel & Lilja, 'Data Prefetch Mechanisms', ACM Computing Surveys 32(2):174-199, 2000, DOI 10.1145/358923.358939). (4) Modern LLM/embedding semantic caches: GPTCache (Zilliz) answers from embedding-space neighborhoods via a similarity threshold over vector stores (Milvus/FAISS/Redis), and 2025-26 research (SISO, arXiv:2508.18736; arXiv:2603.03301 - which proves offline semantic-cache eviction NP-hard) builds locality-aware eviction on top; notably GPTCache itself still evicts with plain LRU/FIFO, which the research literature flags as the open gap. Also SPREE ('Object Prefetching for Mobile Computers') prefetches related objects.

Sources:
- https://courses.cs.duke.edu//spring02/cps296.1/papers/DFJST-VLDB1996.pdf
- https://www.cs.hmc.edu/~geoff/sosp97.html
- https://courses.cs.umbc.edu/graduate/691f/Spring00/Papers/kuenning-sosp97.pdf
- https://dl.acm.org/doi/10.1145/264107.264207
- https://dl.acm.org/doi/pdf/10.1145/358923.358939
- https://zilliz.com/what-is-gptcache
- https://arxiv.org/pdf/2508.18736
- https://arxiv.org/html/2603.03301v1
- https://jilp.org/vol13/v13paper2.pdf

### SUPPORTED: Smarter-than-LRU policies deliver measured hit-rate improvements over plain LRU (so a POC must beat these, not LRU).

Verified numbers from primary abstracts/papers: ARC (Megiddo & Modha, 'ARC: A Self-Tuning, Low Overhead Replacement Cache', FAST 2003; IBM Research abstract): workstation disk trace at 16MB cache, LRU 4.24% vs ARC 23.82% hit ratio (~5.6x); SPC1-like synthetic benchmark at 4GB cache, LRU 9.19% vs ARC 20%; 'substantial gains' across 23 real-life traces; ARC is self-tuning (one adaptation parameter p balancing recency list T1 vs frequency list T2 with ghost lists B1/B2) and was patented by IBM. LRU-2 (O'Neil, O'Neil, Weikum, SIGMOD 1993, pp. 297-306, DOI 10.1145/170035.170081): two-pool experiment (pools of 100 and 10,000 pages) - LRU-2 hit ratio 0.291 at 60 buffers where LRU-1 needs ~140 pages for the same ratio (cost/performance factor ~2.3); Zipfian experiment - more than 2x LRU's hit ratio at buffer sizes under 600. Markov prefetcher (Joseph & Grunwald, ISCA 1997): 54% average reduction in memory-operation execution stalls. Feedback-Directed Prefetching (Srinath, Mutlu, Kim, Patt, HPCA 2007, DOI 10.1109/HPCA.2007.346185): +6.5% average performance over the best static stream-prefetcher configuration on 17 memory-intensive SPEC CPU2000 benchmarks while using 18.7% less bandwidth. GDSF (Cherkasova 1998 + Cherkasova & Ciardo 2001): best file hit ratio among web-proxy policies in independent evaluations (exact deltas live in HPL-98-69's tables, not retrievable this session). LRFU (Lee et al. 2001): 'at least competitive' with prior policies; complexity between O(1) and O(log n). Caution for the POC: ARC's and LRU-K's gains over LRU are largest on scan-heavy/mixed-locality disk workloads and shrink on mildly skewed (Zipfian) workloads - the baseline choice of trace matters enormously.

Sources:
- https://dominoweb.draco.res.ibm.com/d6fe9a14c67ab32e85256ce500621a9a.html
- https://www.usenix.org/legacy/event/fast03/tech/megiddo.html
- https://www.usenix.org/legacy/event/fast03/tech/full_papers/megiddo/megiddo.pdf
- http://www.cs.iit.edu/~yee/classes/cs525aut04/oneil93lruk.pdf
- https://www.cs.cmu.edu/~christos/courses/721-resources/p297-o_neil.pdf
- https://dl.acm.org/doi/10.1145/264107.264207
- http://hps.ece.utexas.edu/pub/srinath_hpca07.pdf

### SUPPORTED: Neighborhood prefetching can HURT: cache pollution is a documented failure mode, with known workload conditions.

Srinath et al. (HPCA 2007) built an entire mechanism around run-time estimation of 'prefetcher-caused cache pollution' because inaccurate prefetches 'displace useful lines that already exist in the cache' and contend with demand requests for memory bandwidth; their fix includes demoting prefetched blocks' insertion position in the L2 LRU stack when pollution is detected. Lee, Kim, Vuduc, 'When Prefetching Works, When It Doesn't, and Why', ACM TACO 9(1), Article 2, March 2012, DOI 10.1145/2133382.2133384: across SPEC CPU 2006, prefetching helped some benchmarks, hurt others, and was neutral on the rest; prefetches issued too early increase cache/bandwidth stress, and software prefetching carries instruction overhead. Seshadri et al., 'Mitigating Prefetcher-Caused Pollution Using Informed Caching Policies for Prefetched Blocks', ACM TACO 2015, extends this and warns that accuracy-only feedback metrics can enter positive feedback loops. Established conditions where prefetching degrades performance: low predictor accuracy (irregular/pointer-chasing/random access), small caches relative to working set (each wrong prefetch evicts a useful line), bandwidth-saturated systems, and overly-early prefetch timing. Direct implication for the hypothesis: 'heating the whole activated neighborhood into cache' is an aggressive prefetch policy; every mispredicted neighbor is pollution, so the POC must measure prefetch accuracy and pollution, not only hit rate, and include a low-semantic-locality (random/adversarial) workload.

Sources:
- http://hps.ece.utexas.edu/pub/srinath_hpca07.pdf
- https://dl.acm.org/doi/pdf/10.1145/2133382.2133384
- https://users.ece.cmu.edu/~omutlu/pub/informed-caching-for-prefetching_taco15.pdf
- https://web.eece.maine.edu/~vweaver/classes/ece571_2022f/ece571_lec12.pdf

### REFUTED: Novelty claim: 'radiating heat to keep related items warm' (activation spreading over a semantic graph with decay) is a new mechanism.

Decay-plus-neighbor-activation is a 50-year-old memory model. Collins & Loftus, 'A Spreading-Activation Theory of Semantic Processing', Psychological Review 82:407-428 (1975): activation spreads outward along network links from primed concepts to linked nodes and onward, attenuating with distance. Anderson's ACT-R architecture formalizes base-level activation driven by recency and frequency of past uses (power-law decay) plus spreading activation from associated context elements. Recent systems work already imports this into computing-memory management (e.g., SYNAPSE, arXiv:2601.02744, spreading activation for LLM agent memory). Combined with SEER's semantic-distance clustering for cache/hoard decisions (SOSP 1997) and Dar et al.'s semantic-locality replacement (VLDB 1996), the 'radiation' component is a recombination of well-documented ideas, not a new mechanism. Note: no prior system was found that implements literal coupled heat-diffusion dynamics (continuous item-to-item temperature transfer) inside a cache eviction loop - that specific formulation may be unoccupied, but its intended behavioral effect is fully anticipated, so the burden on the POC is to show the physics formulation beats ARC/W-TinyLFU/hyperbolic + a Markov or semantic prefetcher, not to claim conceptual novelty.

Sources:
- https://scispace.com/pdf/a-spreading-activation-theory-of-semantic-processing-4e28t5lvm7.pdf
- https://www.semanticscholar.org/paper/A-spreading-activation-theory-of-semantic-Collins-Loftus/61374d14a581b03af7e4fe0342a722ea94911490
- https://arxiv.org/pdf/2601.02744
- https://www.cs.hmc.edu/~geoff/sosp97.html
- https://courses.cs.duke.edu//spring02/cps296.1/papers/DFJST-VLDB1996.pdf

### SUPPORTED: Oracle's buffer cache uses a touch-count (heat-like) algorithm rather than strict LRU.

Multiple independent third-party sources (adp-gmbh.ch Oracle internals page; Jonathan Lewis, 'Oracle Core' OC-5 errata; Oracle internals presentations) agree: each buffer pool's LRU list is split into hot and cold regions (default ~50/50, _db_percent_hot_default); single-block reads use midpoint insertion; each buffer carries a touch count incremented at most once per ~3 seconds (_db_aging_touch_time); buffers with touch count above _db_aging_hot_criteria (default 2) are promoted when a replacement scan passes, with the count reset (Lewis's errata: reset to zero on cold-to-hot move, correcting his book's 'halved'); X$BH exposes the TCH column. Caveat: this is reverse-engineered/practitioner documentation of hidden underscore parameters, not official Oracle docs - Oracle does not publicly document or support tuning these. Characterization: it is a coarse frequency counter with time-gated increments and demotion/reset aging - i.e., another member of the decayed-frequency family the hypothesis claims to invent.

Sources:
- http://www.adp-gmbh.ch/ora/concepts/cache.html
- https://jonathanlewis.wordpress.com/oracle-core/oc-5-caches-and-copies/
- https://arxiv.org/pdf/2512.22995

### SUPPORTED: Redis implements 'LFU with decay' in production, configurable, with lazy sampled eviction.

Official Redis documentation ('Key eviction' page): allkeys-lfu / volatile-lfu policies use a per-key 8-bit Morris-style logarithmic counter; lfu-log-factor (default 10) sets how many accesses saturate the counter at 255; lfu-decay-time (default 1) is 'the amount of minutes a counter should be decayed, when sampled and found to be older than that value'; special value 0 means never decay (several third-party blogs state the opposite - the official doc is authoritative). Eviction is approximate: Redis samples keys rather than maintaining an exact ordering. This is a shipping, widely deployed 'temperature with decay' cache eviction system since Redis 4.0.

Sources:
- https://redis.io/docs/latest/develop/reference/eviction/

Key facts:
- LRFU (Lee et al., IEEE Trans. Computers 50(12):1352-1361, 2001) already formalizes 'temperature' eviction: per-block score C updated as C_new = 1 + F(delta)*C_old with exponential weighing F; lambda sweeps a continuous spectrum from LFU (lambda->0) to LRU; the exponential's F(x+y)=F(x)F(y) property is exactly what makes O(1) 'decay-on-access' updates possible.
- A temperature recurrence W = 1 + W*e^(-alpha*dt) is published verbatim as the EXD eviction policy in tiered-storage literature (arXiv:1907.02394), so 'evict by decayed temperature' is mathematically a reparameterization of LRFU/EXD, not a new objective.
- ARC (Megiddo & Modha, FAST 2003) verified numbers: LRU 4.24% vs ARC 23.82% hit ratio on a workstation trace at 16MB; LRU 9.19% vs ARC 20% on SPC1-like at 4GB; ARC is self-tuning and scan-resistant and was patented by IBM - it is the standard non-LRU baseline a new policy must beat.
- LRU-2 (O'Neil, O'Neil, Weikum, SIGMOD 1993) beat LRU by >2x hit ratio at small buffer sizes on Zipfian access and ~2.3x cost/performance on the two-pool workload - using the time of the Kth-most-recent reference, i.e., frequency-aware eviction predates 'temperature' by three decades.
- Semantic-neighborhood caching exists since 1996: Dar et al. (VLDB 1996) cache 'semantic regions' with constraint descriptions, answer queries partially from cache plus a 'remainder query', and replace regions by semantic distance from the center of gravity of recent queries ('semantic locality' vs temporal locality).
- Semantic-distance-driven whole-cluster caching exists since 1997: SEER (Kuenning & Popek, SOSP 1997) computes pairwise 'semantic distance' between files, clusters them, and hoards entire related clusters, explicitly positioned as superior to LRU-style caching.
- Neighborhood prefetching is standard architecture practice: sequential one-block lookahead (Smith, IEEE Computer, 1978), stride/Reference Prediction Tables (Chen & Baer, IEEE TC 1995), and Markov/correlation prefetchers issuing multiple predicted successors per miss with 54% average memory-stall reduction (Joseph & Grunwald, ISCA 1997); surveyed in VanderWiel & Lilja, ACM Computing Surveys 32(2), 2000.
- Prefetching provably hurts under low accuracy: wrong prefetches pollute the cache (evict useful lines) and burn bandwidth; FDP (Srinath et al., HPCA 2007) gains +6.5% performance / -18.7% bandwidth mainly by THROTTLING prefetch aggressiveness; Lee/Kim/Vuduc (TACO 2012) show SPEC2006 benchmarks where prefetching degrades performance. A 'heat the whole neighborhood' design must therefore report prefetch accuracy and pollution, and be tested on low-locality workloads where it should lose.
- 'No bookkeeping' eviction already ships: Hyperbolic Caching (USENIX ATC 2017) uses lazily-decaying priorities with random 64-item sampling and no ordered structures; Redis LFU does the same with an 8-bit log counter decayed per minute (lfu-decay-time) - both are production/peer-reviewed instances of 'eviction as natural dissipation'.
- Hot/warm/cold 'data temperature' with automatic migration to slower storage is a shipped commercial product (Teradata Virtual Storage, ~2008+), and exponential smoothing for hot/cold record classification was published at ICDE 2013 (Stoica, Levandoski, Larson) with sub-second analysis of 1B-access logs.
- The decay+spreading-activation mechanism is a 1975 cognitive model (Collins & Loftus, Psychological Review 82:407-428) formalized in ACT-R (base-level decay + spreading activation), already being applied to LLM agent memory (SYNAPSE, arXiv:2601.02744).
- Genuinely open edges the POC could legitimately target: GPTCache-style LLM semantic caches still evict with plain LRU/FIFO (semantic-aware eviction is flagged as a gap; offline-optimal semantic caching proven NP-hard in arXiv:2603.03301); and no found system implements literal coupled item-to-item heat diffusion inside the eviction loop - but the bar is beating ARC/W-TinyLFU/hyperbolic + a Markov or semantic prefetcher, not beating plain LRU.
- Access constraint this session: the agent proxy blocked direct PDF downloads (usenix.org, vldb.org, openu.ac.il, duke.edu all CONNECT-403), so formula-level details (LRFU's exact base, Dar's exact Manhattan-distance formula, ARC's full P1-P14 table) were verified through abstracts, snippets, and secondary sources rather than full primary texts; all are flagged in the verdicts where it matters.

## How Teradata and Oracle actually search

Literature review for the thermodynamic-ripple-search POC, from vendor docs and long-standing IR literature. (1) Oracle Text CONTEXT is a classic inverted index with Salton-formula inverse-frequency (TF-IDF-family) scoring scaled 1-100 (vendor-documented since 8i, unchanged through 26ai); CTXCAT is a deprecated short-text variant. (2) Oracle's buffer cache is officially 'LRU lists + touch count' - a scan-resistant midpoint-insertion LRU variant with hot/cold regions. (3) Teradata hash-distributes rows across AMPs; primary-index equality access is a vendor-documented single-AMP hash RETRIEVE whose cost is essentially independent of table size; Teradata has NO inverted full-text index (text LIKE/regex predicates full-scan), and its caching is the FSG segment cache with Cylinder Read and a very-hot tier. (4) The hypothesis's premise that traditional search degrades 'linearly or quadratically' with data size is refuted for all indexed paths: hash ~O(1), B-tree O(log N), inverted-index queries proportional to posting-list lengths (sublinear in corpus for selective terms; worst case linear in matches; never quadratic) - only unindexed scans are O(N), so the benchmark must use indexed baselines to be fair. (5) Teradata Vantage Express is VM-only, registration-gated, unsupported-evaluation-licensed, and impractical in ephemeral containers; Oracle XE/23ai Free has an official FUTC-licensed container image (2 threads/2GB RAM/12GB data caps) and is practical. Recommended open-source proxies: PostgreSQL tsvector+GIN with ts_rank (closest architectural match to Oracle Text), SQLite FTS5 (embedded inverted index + BM25), Lucene (BM25 default since 6.0) as the strong IR baseline, plus a hash-lookup path and an unindexed LIKE scan to proxy Teradata's PI access and its actual (index-less) text-search behavior respectively.

### SUPPORTED: Oracle Text CONTEXT (and CTXCAT) indexes are inverted indexes, and CONTEXT relevance scoring uses an inverse-frequency (TF-IDF family / Salton) algorithm, not BM25

Oracle Text Application Developer's Guide (19c/21c, 'Indexing with Oracle Text') states the indexing engine creates an inverted index that maps tokens to the documents containing them (worked example: token DOG -> DOC1, DOC3, DOC5); tokens are stored in the DR$<index>$I table. The Oracle Text Reference appendix 'The Oracle Text Scoring Algorithm' (present verbatim from 8i through the current 26ai docs) states word-query scoring uses 'an inverse frequency algorithm based on Salton's formula', scores are scaled 1-100 per query, and the number of term occurrences needed to score 100 depends on corpus size (e.g., 34 occurrences in a 1-document set vs 5 in a 1,000,000-document set); DML shifts scores until SYNC/OPTIMIZE. So Oracle Text is a classic inverted index with TF-IDF-style ranking, NOT BM25. CTXCAT is a combined (text + structured columns) index queried with CATSEARCH, intended for short text fragments with structured predicates; it is larger and slower to build than CONTEXT and supports only limited structured operators. Note: CTXCAT is deprecated in recent Oracle releases; CONTEXT is the canonical comparator.

Sources:
- Oracle Corp., Oracle Text Reference 26ai, Appendix E 'The Oracle Text Scoring Algorithm', https://docs.oracle.com/en/database/oracle/oracle-database/26/ccref/oracle-text-scoring-algorithm.html (same text in 18c: https://docs.oracle.com/en/database/oracle/oracle-database/18/ccref/oracle-text-scoring-algorithm.html)
- Oracle Corp., Oracle Text Application Developer's Guide 21c (F31234), ch.3 'Indexing with Oracle Text', https://docs.oracle.com/en/database/oracle/oracle-database/21/ccapp/indexing-with-oracle-text.html
- Oracle Corp., Oracle Text Application Developer's Guide 12.2, https://docs.oracle.com/database/122/CCAPP/indexing-with-oracle-text.htm (CTXCAT/CATSEARCH usage and limits)

### SUPPORTED: Oracle's buffer cache replacement is a touch-count-based LRU variant (hot/cold lists with midpoint insertion), not plain LRU

Current official Oracle 23ai documentation states the database buffer cache 'is managed by a complex algorithm that uses a combination of least recently used (LRU) lists and touch count'. Mechanism details come from well-established internals literature (not official docs): each buffer header carries a touch count (x$bh.tch) incremented at most once per interval (_db_aging_touch_time, default 3s, unlatched so increments can be lost); since 8.1.5 new single-block reads are inserted at a midpoint between a hot and cold region (_db_percent_hot_default ~50%), a buffer reaching the cold tail is relocated rather than evicted if its touch count >= threshold (~2), and multi-block/full-table-scan reads go to the cold/LRU end so scans do not flush the hot working set. Separate KEEP/RECYCLE pools exist. The hidden-parameter specifics are version-dependent and undocumented officially, but the 'LRU lists + touch count' characterization is vendor-confirmed.

Sources:
- Oracle Corp., Oracle Database 23ai documentation, 'Database Buffer Cache' (db_dbbuffercache), https://docs.oracle.com/en/database/oracle/oracle-database/23/dbiad/db_dbbuffercache.html
- Julian Dyke (2005), 'Logical I/O' presentation (touch-count algorithm, midpoint insertion, hidden parameters), mirrored at https://www.scribd.com/presentation/283590996/LogicalIO
- Oracle internals literature summaries (8.1.5+ touch-count/midpoint behavior), e.g. O'Reilly 'Oracle Internals' ch.44, https://www.oreilly.com/library/view/oracle-internals/9780203997536/Part0044.xhtml

### SUPPORTED: Teradata retrieval is built on hash-distributed AMPs, and primary-index equality access is a single-AMP hash lookup (effectively O(1) routing + bounded local lookup), not a scan

Official Teradata Vantage documentation: rows are distributed across AMPs by a hashing algorithm applied to the primary index (PI) columns; a row is stored on the AMP its 32-bit row hash maps to via the hash map (hash bucket -> AMP). When a query supplies values for all PI columns, the PE hashes them and dispatches to exactly one AMP ('Single-AMP Operations Using a Primary Index'); EXPLAIN shows a 'single-AMP RETRIEVE' step, and for a unique PI the row-hash lock is folded into the step. The AMP then locates the row via its file system (master index -> cylinder index -> data block), so access cost is independent of table row count to first order - it is hash routing plus a shallow local structure traversal, categorically not a table scan. Non-PI predicates without a usable secondary/join index DO require an all-AMP full-table scan.

Sources:
- Teradata, VantageCloud Lake Database Reference / Database Design, 'Single-AMP Operations Using a Primary Index', https://docs.teradata.com/r/Teradata-VantageCloud-Lake/Database-Reference/Database-Design/Design-Issues-for-Tactical-Queries/Localizing-the-Work/Single-AMP-Operations-Using-a-Primary-Index
- Teradata, SQL Data Definition Language Syntax and Examples (July 2021), 'PRIMARY INDEX', https://docs.teradata.com/r/SQL-Data-Definition-Language-Syntax-and-Examples/July-2021/Table-Statements/CREATE-TABLE-and-CREATE-TABLE-AS/Syntax-Elements/index/PRIMARY-INDEX
- Teradata, Lake Database Reference, 'Purposes of a Primary Index or Primary AMP Index', https://docs.teradata.com/r/Lake-Database-Reference/Database-Design/Primary-Index-Primary-AMP-Index-and-NoPI-Objects/Primary-Indexes-and-Primary-AMP-Indexes

### SUPPORTED: Teradata has no native inverted-index full-text search; its caching layer is the FSG cache with Cylinder Read for sequential I/O

No Teradata documentation describes a full-text (inverted) index type; DB-Engines lists Teradata secondary indexes as join/aggregate/sparse/hash only, with no full-text index (in contrast to engines that have one). Text predicates in Teradata SQL are LIKE/SUBSTR/POSITION/REGEXP_* evaluated per-row (i.e., scan-based unless a hash/secondary index covers an equality), plus separate analytic-function text tooling (ClearScape/ML text functions, and a vector store for semantic search in recent VantageCloud) - none of which is an inverted keyword index inside the SQL engine. Caching: official docs state the File System Segment (FSG) cache is node-shared memory holding 'the most recently used database segments' - permanent data blocks, cylinder indexes, spool, WAL/transient journal - sized via the ctl utility after OS free-memory guidelines; dirty segments are written on eviction, clean ones dropped. Cylinder Read is the documented mechanism that loads a full cylinder (not block-at-a-time) for sequential operations such as full-table scans, with ResUsageSvpr columns tracking cylinder-read requests and denials when FSG cache is insufficient; a 'very hot' (VH) cache tier keeps the hottest permanent cylinders by temperature (physical+logical access counts). Architecture background (buddy allocation, aging queues) is in Teradata/NCR patent US6505284B1.

Sources:
- Teradata, Vantage Analytics Database (Enterprise/IntelliFlex/VMware) Database Administration, 'Managing FSG Cache', https://docs.teradata.com/r/Enterprise_IntelliFlex_VMware/Database-Administration/Performance-Management-All-DBAs/Performance-and-Memory-Management/Managing-FSG-Cache
- Teradata, Resource Usage Macros and Tables, ResUsageSvpr 'Process Scheduling Cylinder Read Columns', https://docs.teradata.com/r/Enterprise_IntelliFlex_VMware/Resource-Usage-Macros-and-Tables/ResUsageSvpr-Table/Statistics-Columns/Process-Scheduling-Cylinder-Read-Columns
- US Patent 6,505,284 B1 (NCR/Teradata), 'File segment subsystem for a parallel processing database system', https://patents.google.com/patent/US6505284
- DB-Engines, Teradata system entry (no full-text index listed among secondary indexes), https://db-engines.com/en/system/Manticore+Search%3BSphinx%3BTeradata%3BTerminusDB

### REFUTED: 'Traditional engine search slows down linearly or quadratically as data grows' (as applied to Teradata, Oracle, and inverted-index engines)

The claim is wrong for every indexed access path in these systems, and 'quadratically' has no basis anywhere for single-query execution. Precise per-structure behavior: (a) Teradata PI access: hash routing to one AMP + shallow master/cylinder-index lookup - per-lookup cost essentially independent of N (vendor docs describe single-AMP RETRIEVE, no scan). (b) B-tree/B+-tree point lookup: O(log_m N) node accesses - canonical result (Comer 1979; Bayer & McCreight 1972); a billion-row B-tree is ~2x the depth of a thousand-row one, not 1,000,000x. (c) Inverted index (Oracle Text, Lucene, Postgres GIN, FTS5): query cost is proportional to the lengths of the posting lists for the query terms (plus top-k ranking), i.e., scales with the number of matching documents, not corpus size; for selective terms this is strongly sublinear in N (Zobel & Moffat 2006; Manning/Raghavan/Schutze 2008, ch.1-2 and 6). Posting lists do grow roughly linearly with corpus size for a fixed term distribution, so worst-case (very common term, no early-termination) query cost can approach O(matches) ~ O(c*N) with small c - that is the honest upper bound, still never quadratic. (d) The only genuinely linear case is an unindexed scan: LIKE '%term%' on an unindexed column (Teradata without a usable index, Oracle/Postgres without a text/trigram index) is O(N) per query by construction. Any fair benchmark must therefore compare the ripple engine against INDEXED baselines, not naive scans; beating an O(N) scan does not support the hypothesis.

Sources:
- Justin Zobel and Alistair Moffat (2006), 'Inverted Files for Text Search Engines', ACM Computing Surveys 38(2), article 6, https://people.eng.unimelb.edu.au/jzobel/fulltext/compsurv06.pdf
- Christopher D. Manning, Prabhakar Raghavan, Hinrich Schutze (2008), 'Introduction to Information Retrieval', Cambridge University Press (inverted-index query evaluation cost proportional to postings processed)
- Douglas Comer (1979), 'The Ubiquitous B-Tree', ACM Computing Surveys 11(2), pp.121-137 (O(log n) B-tree operations)
- Teradata, 'Single-AMP Operations Using a Primary Index', https://docs.teradata.com/r/Teradata-VantageCloud-Lake/Database-Reference/Database-Design/Design-Issues-for-Tactical-Queries/Localizing-the-Work/Single-AMP-Operations-Using-a-Primary-Index

### REFUTED: Teradata (Vantage Express) can be legally and practically run inside an ephemeral Linux container for a POC benchmark

Teradata Vantage Express is distributed only as a virtual machine image (OVA/VMware disk) via downloads.teradata.com, requiring account registration; it is described as an unsupported evaluation copy ('Teradata Services does not officially support'). Official quickstarts cover VMware, VirtualBox, and UTM only, and require >=1 dedicated core, 6 GB RAM, ~30 GB disk; the cloud quickstarts run the VM via nested virtualization on large instances (AWS bare metal >$3/hr; GCP 4 vCPU/8GB/70GB). There is no official Docker/OCI image of the Teradata database engine, and no redistribution right, so pulling it in CI or an ephemeral container is neither licensed for redistribution nor practical (nested VM inside an unprivileged container will not boot). Teradata's sanctioned free route is the hosted ClearScape Analytics Experience (https://clearscape.teradata.com/), which is a remote service, not a local container. Conclusion for the POC: Teradata itself cannot be a containerized baseline; it must be proxied.

Sources:
- Teradata Downloads, 'Teradata Vantage Express' (VMware), https://downloads.teradata.com/download/database/teradata-express/vmware
- Teradata Developers quickstarts: 'Run Vantage Express on VirtualBox' https://quickstarts.teradata.com/getting.started.vbox ; 'Run Vantage Express on VMware' https://quickstarts.teradata.com/getting.started.vmware.html ; 'Run Vantage Express on AWS' https://quickstarts.teradata.com/run-vantage-express-on-aws.html
- Teradata/quickstarts GitHub repo, getting.started.vmware.adoc, https://github.com/Teradata/quickstarts/blob/main/modules/ROOT/pages/getting.started.vmware.adoc

### SUPPORTED: Oracle XE / Oracle Database Free can be legally and practically run in an ephemeral Linux container

Oracle publishes an official container image: container-registry.oracle.com/database/express:21.3.0-xe (Oracle Database XE 21.3.0 on Oracle Linux 7), licensed under the Oracle Free Use Terms and Conditions (FUTC) bundled in the image - free use including production, within hard caps of 2 CPU threads, 2 GB RAM, 12 GB user data (caps confirmed in the official 21c XE 'Licensing Restrictions' doc). Oracle Database 23ai Free is the official successor to XE (Oracle blog, same free-use terms and essentially the same caps) with image container-registry.oracle.com/database/free. Community images gvenzl/oracle-xe and gvenzl/oracle-free on Docker Hub are slimmed, widely used for CI (podman/docker compatible); gvenzl notes XE images are maintained only for a limited time post-23ai. Practical caveats: multi-GB image, ~1-3 min startup, 2 GB RAM cap constrains benchmark scale. Oracle Text ships with XE/Free per Oracle's XE feature positioning (XE 18c+ advertised as including Text/Spatial/Multimedia), but this specific row was not re-verified in the 21c licensing table during this research - verify empirically at POC setup by checking the CTXSYS schema and creating a CTXSYS.CONTEXT index in the chosen image before relying on it.

Sources:
- Oracle Container Registry, Database Express repository, https://container-registry.oracle.com/ords/ocr/ba/database/express
- Oracle Corp., Oracle Database XE 21c Installation Guide, 'Licensing Restrictions' (2 cores, 12 GB user data), https://docs.oracle.com/en/database/oracle/oracle-database/21/xeinl/licensing-restrictions.html
- Oracle blog, 'Oracle Database 21c XE now Generally Available', https://blogs.oracle.com/database/oracle-database-21c-xe-generally-available
- gvenzl/oci-oracle-xe GitHub (community images; 23c/23ai Free as XE successor), https://github.com/gvenzl/oci-oracle-xe and https://hub.docker.com/r/gvenzl/oracle-xe

### SUPPORTED: PostgreSQL tsvector+GIN, SQLite FTS5, and Lucene are faithful open-source proxies for the commercial systems' text-search architectures

For Oracle Text: PostgreSQL's official docs state GIN is 'the preferred text search index type' and that GIN indexes are inverted indexes containing 'an index entry for each word (lexeme), with a compressed list of matching locations'; ts_rank/ts_rank_cd provide frequency/proximity-based relevance - architecturally the same design as Oracle's CONTEXT (inverted index + TF-style ranking inside an RDBMS, with the same recheck/lossiness caveats). SQLite FTS5 is an inverted-index virtual table with built-in BM25 ranking via the hidden 'rank' column (rank is negative BM25, so ascending ORDER BY rank returns best first) - a faithful embedded-engine proxy. Lucene is the reference open-source inverted index; BM25Similarity has been its default since 6.0 (LUCENE-6789 changed the default and renamed DefaultSimilarity to ClassicSimilarity; Solr followed via SOLR-8261) - the strongest IR baseline, though its BM25 differs from Oracle's Salton-style scoring, so compare rankings, not raw scores. For Teradata: since Teradata has NO full-text index, the faithful proxy is not an FTS engine at all but (a) a hash-based key lookup (e.g., PostgreSQL hash index or in-memory hash table) to model single-AMP PI access, and (b) an unindexed LIKE '%x%' full scan to model Teradata's actual text-predicate behavior. Benchmarks should report both.

Sources:
- PostgreSQL Global Development Group, PostgreSQL documentation, 'Preferred Index Types for Text Search' (textsearch-indexes), https://www.postgresql.org/docs/19/textsearch-indexes.html
- SQLite Consortium, 'SQLite FTS5 Extension', https://sqlite.org/fts5.html (inverted index, bm25() auxiliary function, rank column)
- Apache Lucene JIRA LUCENE-6789 'Switch Similarity default to BM25' (fixed in 6.0; DefaultSimilarity renamed ClassicSimilarity), https://issues.apache.org/jira/browse/LUCENE-6789 and SOLR-8261 https://jira.apache.org/jira/browse/SOLR-8261
- Oracle Text scoring appendix (for the ranking-family comparison), https://docs.oracle.com/en/database/oracle/oracle-database/26/ccref/oracle-text-scoring-algorithm.html

Key facts:
- Oracle Text CONTEXT is an inverted index (token -> posting list, stored in DR$<index>$I); relevance for word queries uses 'an inverse frequency algorithm based on Salton's formula' (TF-IDF family), scores scaled 1-100 per query - it is NOT BM25 (Oracle Text Reference, scoring appendix, 8i through 26ai).
- CTXCAT is a combined text+structured index queried with CATSEARCH, for short text fragments; larger/slower to build than CONTEXT and deprecated in recent releases - use CONTEXT as the Oracle comparator.
- Oracle 23ai docs officially describe buffer cache replacement as 'a combination of LRU lists and touch count'; internals literature details midpoint insertion into hot/cold regions, tch counter throttled by _db_aging_touch_time (3s default), full-scan blocks placed at the cold end - a scan-resistant LRU variant, not plain LRU.
- Teradata rows are hash-distributed across AMPs by primary-index value; a query supplying all PI values is dispatched to exactly one AMP (single-AMP RETRIEVE in EXPLAIN) - per-lookup cost essentially independent of table size; it is not a scan.
- Teradata has no inverted/full-text index type; text predicates (LIKE, REGEXP_*) without a usable index run as all-AMP full-table scans - O(N). Its cache is the node-shared FSG cache (most-recently-used file-system segments; ctl-sized), with Cylinder Read loading whole cylinders for sequential scans and a very-hot cache tier by cylinder temperature.
- The hypothesis claim 'traditional engines slow linearly or quadratically with data growth' is REFUTED for indexed paths: hash access ~O(1), B-tree O(log N) (Comer 1979), inverted-index query cost proportional to posting-list lengths of query terms (Zobel & Moffat 2006, ACM Comput. Surv. 38(2)) - sublinear in corpus size for selective queries. Linear holds ONLY for unindexed scans (LIKE '%x%'); quadratic holds nowhere for single-query execution.
- Benchmark design consequence: the POC must compare against indexed baselines (GIN, FTS5, Lucene) AND report the unindexed-scan baseline separately; beating an O(N) scan is not evidence against sublinear indexed engines.
- Teradata Vantage Express: VM-image only (OVA for VMware/VirtualBox/UTM), registration-gated, unsupported evaluation copy, needs >=1 core/6GB RAM/~30GB disk, no official Docker image, no redistribution right - NOT practical/licensed for an ephemeral Linux container. Sanctioned free alternative is the hosted ClearScape Analytics Experience (clearscape.teradata.com).
- Oracle XE 21c IS containerizable legally and practically: official image container-registry.oracle.com/database/express:21.3.0-xe under Oracle Free Use Terms and Conditions; caps 2 CPU threads / 2 GB RAM / 12 GB user data; Oracle Database 23ai Free is the successor (gvenzl/oracle-free community images for CI). Verify Oracle Text (CTXSYS) is usable in the chosen image before benchmarking.
- Faithful open-source proxies: PostgreSQL tsvector+GIN (official docs: GIN is an inverted index, 'preferred text search index type'; ts_rank TF-style ranking) is the closest analog to Oracle Text CONTEXT; SQLite FTS5 (inverted index + BM25, rank = negative BM25 so ascending sort = best first) for an embedded engine; Lucene (BM25 default since 6.0, LUCENE-6789) as the strongest IR baseline. Proxy for Teradata PI access is a hash lookup (e.g., Postgres hash index), and for Teradata text search an unindexed LIKE scan - because Teradata has no FTS.

## Benchmark methodology and state-of-the-art numbers

The literature has a settled answer. Fair search/ANN benchmarking follows the ANN-Benchmarks protocol (Aumüller, Bernhardsson & Faithfull, Information Systems 87, 2020; SISAP 2017): every algorithm gets a parameter sweep, results are plotted as recall@k vs queries-per-second, and only the Pareto frontier is compared — a latency number without its recall is meaningless because approximate indexes can trade recall for arbitrary speed. The protocol runs each implementation single-CPU in Docker, in-memory (warm), best of 5 runs, with index build time and size recorded separately from query throughput; the NeurIPS'21 billion-scale challenge (Simhadri et al., PMLR 176) uses the same recall-at-fixed-throughput ranking. SOTA context for a POC: HNSW (Malkov & Yashunin, TPAMI 2020) delivers roughly 10^3-10^4 QPS per core at recall@10 of 0.9-0.99 on million-scale datasets (verified datapoint: hnswlib ~8,300 QPS at recall 0.897 on glove-100-angular, single thread), and on SIFT1M essentially all modern methods reach near-perfect recall. The lexical baseline is equally pinned: Anserini/Lucene BM25 on MS MARCO passage (8.84M passages) gives MRR@10 0.187 at ~21-50 ms/query single-threaded. Documented pitfalls to avoid: comparing implementations rather than algorithms (Kriegel et al., KAIS 2017), JVM/JIT warmup contaminating Lucene measurements (Georges et al., OOPSLA 2007), Python interpreter overhead (use native/batch APIs), and comparing exact vs approximate methods at unmatched recall. For a small honest POC: use ANN_SIFT10K/SIFT1M (TEXMEX, exact L2 ground truth shipped), ann-benchmarks GloVe HDF5 files (top-100 true neighbors shipped), and 20 Newsgroups with headers/footers/quotes stripped; compute any missing ground truth by brute-force exact k-NN and report recall-vs-QPS curves plus latency percentiles against tuned HNSW and brute force on identical hardware, single-threaded, with build time reported separately. Caveats: ann-benchmarks is now unmaintained (successor: VIBE, arXiv:2505.17810); percentile latency reporting comes from the systems literature (Dean & Barroso, CACM 2013), not from ann-benchmarks itself, which reports QPS; and exact QPS figures are hardware-dependent, so cross-paper numbers are indicative, not comparable.

### SUPPORTED: The accepted methodology for benchmarking ANN algorithms is the ANN-Benchmarks protocol (Aumüller, Bernhardsson, Faithfull): sweep each algorithm's parameters, plot recall vs queries-per-second, and compare only points on the Pareto (precision-performance) frontier.

Aumüller, Bernhardsson & Faithfull, 'ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms', SISAP 2017 (LNCS 10609, pp. 34-49) and journal version in Information Systems 87 (2020), 101374, DOI 10.1016/j.is.2019.02.006. The tool's configuration system 'automatically tests a range of parameter settings for each algorithm'; the official README instructs to 'Try many different values of parameters for each library and ignore the points that are not on the precision-performance frontier.' The NeurIPS'21 billion-scale challenge (Simhadri et al., PMLR 176) uses the same principle, ranking by recall at a fixed query-throughput threshold (10,000 QPS).

Sources:
- Aumüller M., Bernhardsson E., Faithfull A. (2020) 'ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms', Information Systems 87:101374, DOI 10.1016/j.is.2019.02.006 — https://www.sciencedirect.com/science/article/abs/pii/S0306437918303685
- arXiv:1807.05614 — https://arxiv.org/pdf/1807.05614
- https://github.com/erikbern/ann-benchmarks (README)
- Simhadri et al. (2022) 'Results of the NeurIPS'21 Challenge on Billion-Scale Approximate Nearest Neighbor Search', PMLR 176 — https://proceedings.mlr.press/v176/simhadri22a.html

### SUPPORTED: Reporting latency/QPS without recall is meaningless for approximate search; speed and accuracy must be reported as a joint tradeoff.

This is the organizing principle of both major benchmarks: ANN-Benchmarks compares algorithms only on the recall-vs-QPS frontier, and the NeurIPS'21 challenge ranked submissions 'based on recall at a query throughput threshold' (T1 leaderboard: recall achieved at 10,000 QPS on an Azure F32v2). An approximate index can trade recall for arbitrary speed (e.g. hnswlib on glove-100-angular: ~8,300 QPS at recall 0.897 but ~880 QPS at 0.99 in the LUCENE-9937 run), so a QPS number alone is uninterpretable.

Sources:
- https://github.com/erikbern/ann-benchmarks
- https://proceedings.mlr.press/v176/simhadri22a.html
- Apache Lucene JIRA LUCENE-9937 'ann-benchmarks results for HNSW search' — https://issues.apache.org/jira/browse/LUCENE-9937

### SUPPORTED: Index build time must be measured and reported separately from query time.

ANN-Benchmarks records index build time per run as a separate metric from query throughput (the tool caps each parameter setting at 'five hours to build the index and answer the queries' and the README's principles include 'Avoid extremely costly index building (more than several hours)'); the site publishes separate build-time-vs-recall plots. The NeurIPS'21 challenge likewise constrained index build resources separately from the query-phase measurement.

Sources:
- arXiv:1807.05614 — https://arxiv.org/pdf/1807.05614
- https://github.com/erikbern/ann-benchmarks
- https://proceedings.mlr.press/v176/simhadri22a.html

### MIXED: The standard benchmarks enforce an explicit warm-vs-cold cache discipline.

ANN-Benchmarks does not have a cold-cache protocol; it is a deliberately warm, in-memory, best-case measurement: all datasets fit in RAM, and the maintainer states 'We run every algorithm 5 times and pick the best run' (erikbern, ann-benchmarks issue #363, April 2023 run). Best-of-5 explicitly measures warm steady-state and discards warmup-affected runs. Cold-cache/IO effects become a first-class concern only for disk-resident indexes: DiskANN (Jayaram Subramanya et al., NeurIPS 2019) benchmarks SSD-based search (SIFT1B: >5,000 QPS, mean latency <3 ms, >95% 1-recall@1 on 64GB RAM + SSD), where page-cache state materially changes results. For an in-memory POC the honest discipline is: state whether numbers are warm, run repeated passes, and report after warmup.

Sources:
- https://github.com/erikbern/ann-benchmarks/issues/363
- Jayaram Subramanya S. et al. (2019) 'DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node', NeurIPS 2019 — https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html

### SUPPORTED: Results must declare single-thread vs multi-thread execution; the canonical ANN-Benchmarks numbers are single-CPU.

README: 'ANN-Benchmarks enforces that only one CPU is saturated during experimentation, i.e., no multi-threading' and 'All benchmarks are single-CPU'; a separate --batch mode exists for implementations that parallelize across queries. Official runs: AWS r6i.16xlarge, hyperthreading disabled, --parallelism 31 (31 concurrent single-CPU experiments, each algorithm still confined to one core). The IR side does the same: Anserini's 2023 dense-retrieval paper reports single-thread and 12/16-thread throughput in separate columns, and the ColBERT paper (Khattab & Zaharia, SIGIR 2020) explicitly states it reports single-threaded latency for BM25/Anserini.

Sources:
- https://github.com/erikbern/ann-benchmarks
- https://github.com/erikbern/ann-benchmarks/issues/363
- Lin et al., 'Anserini Gets Dense Retrieval: Integration of Lucene's HNSW Indexes' — https://arxiv.org/pdf/2304.12139
- Khattab & Zaharia (2020) 'ColBERT', SIGIR 2020 — https://arxiv.org/pdf/2004.12832

### MIXED: Percentile (tail) latencies, not means, are the accepted way to report query latency.

In serving-systems literature this is settled: Dean & Barroso, 'The Tail at Scale', CACM 56(2):74-80 (2013) shows means hide tails (a 1-in-100 slow server makes 63% of 100-way fan-out requests slow). However, the canonical ANN academic benchmarks report throughput (QPS), not percentiles: ANN-Benchmarks plots QPS vs recall, and the DiskANN NeurIPS 2019 abstract reports *mean* latency (<3 ms); p99 figures circulating for DiskANN are from secondary blogs, not the paper. Honest POC practice: report the full latency distribution (p50/p95/p99 and mean) plus QPS-at-recall; claiming 'percentiles are required by ann-benchmarks' would be false.

Sources:
- Dean J., Barroso L.A. (2013) 'The Tail at Scale', Communications of the ACM 56(2):74-80 — https://research.google/pubs/pub40801/
- https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html
- https://github.com/erikbern/ann-benchmarks

### SUPPORTED: HNSW is state-of-the-art on million-scale benchmarks, reaching recall@10 >= 0.9 at thousands of QPS single-threaded (SIFT1M, GloVe).

Algorithm: Malkov & Yashunin, 'Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs', IEEE TPAMI 42(4) (published online Dec 2018, issue 2020; arXiv:1603.09320). The ANN-Benchmarks paper reports HNSW is fastest over all recall values on GloVe and that on SIFT 'all tested algorithms can achieve close to perfect recall'. Concrete single-thread datapoints: hnswlib on glove-100-angular (1.18M vectors, 100-d, angular; M=32, efConstruction=500): recall@10 0.897 at ~8,300 QPS; 0.981 at ~1,932 QPS; 0.99 at ~880 QPS (LUCENE-9937 ann-benchmarks run, 2021, third-party hardware). Vicinity single-thread GloVe-25 (1.18M vectors): HNSW M=16 at 100% recall@10, 2,857 QPS. Exact QPS varies 2-10x with hardware and parameters; the robust claim is order 10^3-10^4 QPS/core at recall 0.9-0.99 on ~1M points. Official current curves: ann-benchmarks.com per-dataset pages (r6i.16xlarge runs).

Sources:
- Malkov Y.A., Yashunin D.A. (2020) IEEE TPAMI 42(4):824-836, arXiv:1603.09320 — https://arxiv.org/pdf/1603.09320
- https://issues.apache.org/jira/browse/LUCENE-9937
- arXiv:1807.05614 (ANN-Benchmarks paper)
- https://ann-benchmarks.com/glove-100-angular_10_angular.html
- https://docs.rs/crate/vicinity/0.9.1

### SUPPORTED: BM25/Lucene baseline on MS MARCO passage: MRR@10 ~0.187 with per-query latency in the tens of milliseconds single-threaded.

Effectiveness is precisely reproducible: Anserini's official MS MARCO passage doc reports MRR@10 = 0.18741 with tuned BM25 (k1=0.82, b=0.68) and 0.1840 with Lucene defaults (k1=0.9, b=0.4) on the 6,980 dev queries over 8,841,823 passages; the Pyserini paper (Lin et al., SIGIR 2021) confirms 0.1874. Indexing the full 8.8M-passage corpus takes ~3 minutes on an M2 MacBook Air (Anserini docs). Latency is hardware-dependent and reported by third parties: 21 ms/query (EPIC paper, MacAvaney et al. 2020, arXiv:2004.14245, BM25-from-Anserini row), ~50 ms/query (RepBERT, arXiv:2006.15498); ColBERT (SIGIR 2020) also reports single-threaded Anserini BM25 latency in this range. So: tens of ms/query at k=1000, single thread, is the defensible baseline figure.

Sources:
- https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md
- Lin J. et al. (2021) 'Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations', SIGIR 2021 — https://cs.uwaterloo.ca/~jimmylin/publications/Lin_etal_SIGIR2021_Pyserini.pdf
- MacAvaney et al. (2020) 'Expansion via Prediction of Importance with Contextualization' — https://arxiv.org/pdf/2004.14245
- Zhan et al. (2020) 'RepBERT' — https://arxiv.org/pdf/2006.15498
- Khattab & Zaharia (2020) ColBERT — https://arxiv.org/pdf/2004.12832

### SUPPORTED: Known benchmarking pitfalls — measuring implementation/language overhead instead of the algorithm, JIT/VM warmup, and comparing exact vs approximate methods without recall parity — are documented in the peer-reviewed literature.

(1) Implementation-vs-algorithm confound: Kriegel, Schubert & Zimek (2017) 'The (black) art of runtime evaluation: Are we comparing algorithms or implementations?', Knowledge and Information Systems 52(2):341-378, DOI 10.1007/s10115-016-1004-2, shows runtime conclusions flip with implementation language/quality and indexing. ANN-Benchmarks mitigates Python-wrapper overhead with Docker-isolated native libraries and a --batch mode that hands all queries to the implementation at once. (2) JIT warmup: Georges, Buytaert & Eeckhout (2007) 'Statistically Rigorous Java Performance Evaluation', OOPSLA 2007, pp. 57-76, shows prevalent methodologies 'can even lead to incorrect conclusions' due to JIT compilation, GC and scheduling, and prescribes separate startup vs steady-state measurement with confidence intervals — directly applicable to Lucene/JVM baselines (the LUCENE-9937 thread's first hnswlib-vs-Lucene comparison had to be corrected for a run error). (3) Recall parity: ANN-Benchmarks' frontier methodology exists precisely because exact and approximate methods are only comparable at matched recall.

Sources:
- Kriegel H.-P., Schubert E., Zimek A. (2017) KAIS 52(2):341-378, DOI 10.1007/s10115-016-1004-2
- Georges A., Buytaert D., Eeckhout L. (2007) OOPSLA 2007:57-76 — https://dri.es/files/oopsla07-georges.pdf
- https://github.com/erikbern/ann-benchmarks
- https://issues.apache.org/jira/browse/LUCENE-9937

### SUPPORTED: 20 Newsgroups, SIFT subsets (SIFT10K/SIFT1M), and GloVe vectors are standard public datasets suitable for a small honest POC, with established ground-truth conventions.

20 Newsgroups: ~18k posts in 20 topics (Lang 1995, 'NewsWeeder: Learning to filter netnews', ICML); scikit-learn's official docs warn that headers/signatures/quoted replies (e.g. NNTP-Posting-Host) let classifiers overfit metadata and recommend remove=('headers','footers','quotes') — 'the F-score will be lower because it is more realistic'. SIFT: TEXMEX corpus (corpus-texmex.irisa.fr; Jégou, Douze & Schmid, 'Product Quantization for Nearest Neighbor Search', IEEE TPAMI 33(1), 2011) provides ANN_SIFT10K (10K base) and ANN_SIFT1M (1,000,000 base vectors, 128-d, 10,000 queries, .fvecs/.ivecs format) with a groundtruth file of exact L2 nearest neighbors (100 per query). GloVe: Pennington, Socher & Manning (2014) 'GloVe: Global Vectors for Word Representation', EMNLP 2014; ann-benchmarks packages glove-25/50/100/200-angular as HDF5 with 1,183,514 train vectors, 10,000 test queries, and precomputed top-100 true neighbors.

Sources:
- https://scikit-learn.org/stable/datasets/real_world.html (20 newsgroups section)
- Jégou H., Douze M., Schmid C. (2011) IEEE TPAMI 33(1):117-128; dataset at http://corpus-texmex.irisa.fr/
- https://www.tensorflow.org/datasets/catalog/sift1m
- Pennington, Socher, Manning (2014) EMNLP — https://nlp.stanford.edu/projects/glove/
- https://github.com/erikbern/ann-benchmarks (dataset table)

### SUPPORTED: Researchers compute ANN ground truth by exact brute-force k-NN over the full base set for each query, distributed as precomputed neighbor lists.

All three major benchmark families ship precomputed exact neighbors: ann-benchmarks HDF5 datasets are 'pre-split into train and test sets, with precomputed top-100 nearest neighbors'; TEXMEX SIFT1M ships an .ivecs groundtruth file with 100 exact L2 nearest neighbors per query; the NeurIPS'21 billion-scale organizers released ground truth computed by exact search for their six datasets. Recall@k is then |returned ∩ true top-k| / k against those lists. Tools fall back to brute force when no file exists (e.g. GGNN docs: 'the ground truth will be brute-forced, if possible'). For a POC: compute exact top-k once with a brute-force pass (e.g. faiss IndexFlat or numpy) and cache it; never use an approximate index as ground truth.

Sources:
- https://github.com/erikbern/ann-benchmarks
- http://corpus-texmex.irisa.fr/
- https://proceedings.mlr.press/v176/simhadri22a.html
- https://ggnn.readthedocs.io/en/latest/_sources/benchmarking.rst.txt

### MIXED: ann-benchmarks is the current actively-maintained canonical benchmark.

It remains the canonical methodology and citation, but the official README now states the project is 'no longer actively maintained' and points to VIBE (Vector Index Benchmark for Embeddings, arXiv:2505.17810, built on the ANN-Benchmarks pipeline, 21 implementations, 12 in-distribution + 6 out-of-distribution datasets) as a successor. The last official full run used an AWS r6i.16xlarge with 31 parallel single-CPU jobs, best-of-5 runs per algorithm.

Sources:
- https://github.com/erikbern/ann-benchmarks
- https://arxiv.org/pdf/2505.17810
- https://github.com/erikbern/ann-benchmarks/issues/363

Key facts:
- Canonical citation: Aumüller, Bernhardsson, Faithfull, 'ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms', Information Systems 87 (2020) 101374, DOI 10.1016/j.is.2019.02.006 (conference version SISAP 2017; arXiv:1807.05614).
- Core protocol: sweep parameters per algorithm, plot recall@k vs QPS, keep only the Pareto frontier; never report latency or QPS without the recall it was achieved at.
- ann-benchmarks run discipline: Docker-isolated implementations, one CPU saturated per run ('All benchmarks are single-CPU'), datasets in RAM (warm), each algorithm run 5 times keeping the best run, 5-hour cap on build+query per configuration; official hardware AWS r6i.16xlarge, hyperthreading disabled, 31 parallel jobs.
- Index build time and index size are recorded and reported separately from query throughput.
- HNSW (Malkov & Yashunin, IEEE TPAMI 42(4), arXiv:1603.09320) is the de facto SOTA graph index: single-thread hnswlib on glove-100-angular (1.18M x 100d) hits recall@10 ~0.90 at ~8,300 QPS and ~0.98 at ~1,900 QPS (M=32, efC=500, LUCENE-9937 run); order 10^3-10^4 QPS/core at recall 0.9-0.99 on ~1M vectors; on SIFT1M nearly all modern algorithms reach near-perfect recall, differing only in speed.
- BM25 baseline (Anserini/Lucene, MS MARCO passage: 8,841,823 passages, 6,980 dev queries): MRR@10 = 0.1874 tuned (k1=0.82, b=0.68), 0.1840 default (k1=0.9, b=0.4); single-thread latency ~21-50 ms/query at k=1000 (EPIC 21 ms; RepBERT 50 ms); full-corpus indexing ~3 min on an M2 laptop.
- Latency reporting: academic ANN benchmarks report QPS-at-recall; production/serving literature requires percentiles (Dean & Barroso, 'The Tail at Scale', CACM 56(2) 2013: mean hides tails; 100-way fan-out turns a 1% slow tail into 63% of requests). Honest POC: report p50/p95/p99 AND QPS-at-recall.
- Pitfall sources: Kriegel/Schubert/Zimek, KAIS 52(2):341-378 (2017) — runtime comparisons often measure implementations, not algorithms; Georges/Buytaert/Eeckhout, OOPSLA 2007 — JVM/JIT warmup requires separate startup vs steady-state measurement (applies to any Lucene baseline); mitigate Python interpreter overhead via native libraries + batch query APIs (ann-benchmarks --batch).
- Ground truth = exact brute-force k-NN per query, precomputed and cached: TEXMEX SIFT1M ships 100 exact L2 neighbors per query (.ivecs); ann-benchmarks HDF5 ships top-100 true neighbors; never use an approximate method as ground truth; recall@k = |returned ∩ true top-k| / k.
- POC-suitable datasets: ANN_SIFT10K (10K vectors) and ANN_SIFT1M (1M x 128d, 10K queries) from corpus-texmex.irisa.fr (Jégou et al., TPAMI 2011); glove-25/50/100/200-angular HDF5 from ann-benchmarks (1,183,514 train / 10,000 test, GT included; GloVe: Pennington et al., EMNLP 2014); 20 Newsgroups (~18k docs, 20 classes, Lang 1995) for text/BM25 — MUST use remove=('headers','footers','quotes') per scikit-learn docs or results are inflated by metadata leakage.
- ann-benchmarks itself is no longer actively maintained; its README points to VIBE (arXiv:2505.17810) as the maintained successor, built on the same pipeline — cite the methodology from the 2020 paper either way.
- Any benchmark claim for a new method ('thermodynamic ripple' or otherwise) must therefore show: (a) a recall-vs-QPS curve against tuned HNSW and brute-force on the same machine, single-threaded, same k; (b) separate build time; (c) warm steady-state numbers after warmup with repeats; (d) exact ground truth — a bare latency win at unmatched recall is not evidence.
