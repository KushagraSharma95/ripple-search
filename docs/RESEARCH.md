# Research Results: Calibrated Adaptive Termination (exp11–exp13)

Three research questions grew out of the Part 3 controller, each run under a
frozen protocol on fresh query seeds with registered predictions, published
win or lose. Raw numbers: `results/exp11_conformal_contracts.json`,
`exp12_adaptivity_theory.json`, `exp13_transfer.json`.

## 1. Distribution-free recall guarantees work — and are sometimes free (exp11)

**Question.** Can the empirical "mean recall ≥ T, usually" contract be
upgraded to a per-query, distribution-free guarantee?

**Method.** Running recall during beam search is monotone in the stop
threshold, so each calibration query has a well-defined critical threshold
`c_i` — the smallest stop threshold under which it meets an m-of-10
contract (or +∞ if even the full budget never does). Split conformal
calibration takes the ⌈(n+1)(1−δ)⌉-th smallest `c_i` as the deployed
threshold; by exchangeability a fresh query meets the contract with
probability ≥ 1−δ. Contracts are integer (m ∈ {8, 9, 10} of 10) to avoid
granularity ambiguity; a contract whose calibration infeasibility exceeds δ
is declared unattainable rather than faked. Fresh seed 311; 600/300/300
train/calibration/test; baseline = cheapest fixed ef whose *calibration*
coverage clears 1−δ, evaluated once on test.

**Results.** The guarantee held in **12 of 12 cells** (test coverage
0.883–0.983 against targets 0.8/0.9). The unguaranteed fixed baseline
**violated its coverage promise in 5 of 12 cells** — all on Reuters, e.g.
promising 90% and delivering 87.7–89.3%. The price of the guarantee:
*negative* on the synthetic corpus — conformal was cheaper AND guaranteed
in all 6 cells (up to 23% less work than the coverage-matched fixed
budget) — versus 1.6–2.0× extra work on Reuters, where per-query dynamics
are noisier and conformal pays with conservatism. Takeaway: validity is
universal; the cost of validity is corpus-dependent, and the baseline you
would use instead fails silently precisely where the guarantee is
expensive.

## 2. When adaptivity pays: the difficulty-variance theory, amended (exp12)

**Question.** Part 3 measured the controller's dividend growing with the
declared target and vanishing on homogeneous queries. Registered
prediction: the dividend is driven by query-difficulty variance — rising
with spread at every target, ~0 at zero variance.

**Method.** One fixed substrate (synthetic 100k, d=64); five query
difficulty distributions with matched mean noise (eps mean 0.08) and
increasing spread: constant, U[0.06,0.10], U[0.03,0.13], U[0.01,0.15],
and bimodal {0.02, 0.14}. Fresh seed 419. Dividend = work saved vs the
val-selected cheapest sufficient fixed budget, per (distribution, target).

**Results.** Half confirmed, half refuted — and the refuted half is the
finding.

| spread → | const | narrow | medium | wide | bimodal |
|---|---|---|---|---|---|
| T=0.90 | +13.2% | +12.0% | +19.7% | +24.3% | +21.0% |
| T=0.95 | +7.8% | +17.6% | +21.7% | +26.9% | +33.1% |
| T=0.99 | +10.2% | +15.2% | +10.7% | +10.4% | **+35.9%** |

The monotone trend in injected spread holds at 0.90/0.95. But the
registered "~0 at zero variance" is **refuted**: constant-noise queries
still yield +8–13%, because per-query difficulty varies *intrinsically*
with the corpus region the query lands in (local density, hubness) even at
identical noise. The amended theory: the adaptive dividend tracks **total
realized difficulty variance**, of which injected spread is only one
component — fixed budgets must cover the hard tail from *any* source.
Caveats, stated plainly: difficulty is nonlinear in eps, so matched-mean
eps does not perfectly match mean difficulty across distributions (bimodal
selected smaller baseline budgets); dividends are within-distribution
comparisons, which limits but does not remove this confound. Several
controller cells sit a few tenths of a point under the mean-recall target
(e.g. 0.8977 vs 0.90), as do some baselines; contract-met flags are in the
JSON.

## 3. The controller does not transfer across corpora (exp13)

**Question.** Is per-corpus training necessary, or do search-dynamics
features generalize? Registered prediction: scale-free features (ratios,
similarities, budget fractions) transfer better than raw counts.

**Results.** A clear negative. Transferred predictors with local tau
calibration overshoot badly: synthetic→Reuters dividends of **−112% to
−232%** (3–4× the native controller's work, worse than the plain fixed
baseline) versus native +17–22%; Reuters→synthetic transfer roughly ties
the fixed baseline (+2–3%) versus native +25–29%. The registered
prediction is **refuted**: the scale-free subset transferred *worse* in
one direction (−23% to −31%). Conclusion: dynamics features are
corpus-conditional; per-corpus training — cheap (~600 labeled queries,
one instrumented pass) but mandatory — is part of the method's honest
deployment story. Corpus-statistics-conditioned predictors are the obvious
future work.

## What this adds up to

A calibrated, adaptively terminated search layer with (1) distribution-free
per-query recall guarantees, validated 12/12 against a default that fails
silently 5/12; (2) a predictive, partially-corrected theory of when the
approach pays (total difficulty variance × target height); (3) a measured
deployment boundary (no cross-corpus transfer). The guarantee mechanism is,
to our knowledge from the 2024–26 literature sweep in FRONTIER.md, not
present in the DARTH/LAET/Ada-ef line, which reports empirical
target-hitting only. Next steps that would make this external-grade: a
hierarchical substrate so GloVe-class corpora and the 0.999 regime open
up, the same harness inside a production graph library for wall-clock
confirmation, and conformal variants that reduce the Reuters-style
conservatism (e.g. Mondrian/conditional calibration by predicted
difficulty bucket).
