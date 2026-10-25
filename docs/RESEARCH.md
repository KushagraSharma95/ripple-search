# Research Results: Calibrated Adaptive Termination (exp11–exp13)

Three research questions grew out of the Part 3 controller. Each was run
under a protocol frozen before its results existed, with registered
predictions, and published win or lose. Raw numbers:
`results/exp11_conformal_contracts*.json`, `exp12_adaptivity_theory*.json`,
`exp13_transfer*.json`.

**How to read the numbers.** Each experiment has two independent query
seeds. Each seed uses a 300-query test set, so a single coverage or recall
cell carries roughly ±1.7–2.3 points of binomial noise. Conclusions rest on
patterns that hold across cells and seeds, never on one cell.

**Review history.** An adversarial review of the first draft confirmed 18
of 23 findings, one of them critical. The stop-threshold calibration
(bisection over [T−0.05, 1.10]) had an artificial floor. It pinned 6 exp10
cells, 2 exp12 cells and every synthetic→Reuters exp13 cell. In each of
those the controller overshot its target and spent more work than
necessary, which understated its savings and exaggerated exp13's transfer
failure. The corrected search covers [−0.5, 1.5] in 14 steps and flags
any cell still pinned at the floor. The second seeds (exp11: 809, exp12:
701, exp13: 617) were registered in git before running. First-seed cells
affected by the floor are marked below and are not used as evidence for
claims they would inflate.

## 1. Distribution-free recall guarantees (exp11): 24 of 24

**Question.** Can "mean recall ≥ T, usually" become a per-query,
distribution-free guarantee?

**Method.** During beam search, the running recall at the stop point never
decreases as the stop threshold rises. So each calibration query has a
critical threshold `c_i`: the smallest threshold at which it meets an
m-of-10 contract, or +∞ if even the full budget never does. Split
conformal calibration deploys the ⌈(n+1)(1−δ)⌉-th smallest `c_i`. By
exchangeability, a fresh query then meets the contract with probability
≥ 1−δ. Contracts are integer (m ∈ {8, 9, 10} of 10). If more than a δ
fraction of calibration queries are infeasible, the contract is declared
unattainable. The design is 600 train / 300 calibration / 300 test. The
baseline is the cheapest fixed ef whose calibration coverage clears 1−δ,
evaluated once on test.

**Results.**

| | conformal (both seeds) | naive fixed baseline (both seeds) |
|---|---|---|
| coverage promise kept | **24/24 cells** | 14/24 cells |
| Reuters test coverage | 0.877–0.983 | violated in 9/12 cells |
| synthetic test coverage | 0.820–0.963 | violated in 1/12 cells |

The baseline's misses are 0.3–4 points each, and most are individually
within about 2 standard errors. The aggregate is the evidence: 10 of 24 is
what a rule with no safety margin produces. Selecting "the cheapest config
that clears the bar on calibration" picks the config that got lucky on
calibration, so it lands near the boundary and fails on test about half
the time. This is the selection bias conformal calibration is built to
absorb.

**Price of the guarantee.**
- **Synthetic:** conformal was cheaper *and* guaranteed in 12/12 cells,
  with 6.6–24.9% less work than the baseline.
- **Reuters:** conformal spent 1.08–1.93× the baseline's work, and the
  price depended on the seed (1.08–1.41× on seed 809, 1.58–1.93× on
  seed 311).

Part of the Reuters "price" is simply the cost of actually keeping a
promise that the cheaper baseline breaks in 9 of 12 cells.

**Disclosures.**
- *Conservative bug in seed 311.* `critical_value` returned +∞ for a
  calibration trace with no checkpoints even when the full budget met the
  contract; the correct value is −∞. That error can only raise the
  deployed threshold, so seed 311's validity is unaffected. Seed 809 ran
  with the corrected function.
- *Dropped baseline.* The registered context baseline (exp10's
  mean-recall controller) was never implemented. It is dropped here, with
  this note, because it guarantees nothing per query and exp10 reports it
  on its own seeds.
- *Stale protocol text in one result file.* The scripted docstring
  amendment for seed 809 was a silent no-op (whitespace mismatch), so
  `exp11_conformal_contracts_seed809.json` embeds the v1 description. Its
  `query_seed` field (809) is authoritative. The amendment is now in the
  code, and the seed was registered in commit ec1b29d before the run.

## 2. When adaptivity pays: the difficulty-variance theory (exp12)

**Question.** Registered prediction: the controller's dividend is driven
by query-difficulty variance. It should rise with spread at every target
and be about zero when variance is zero.

**Method.**
- **Substrate:** one fixed corpus, synthetic 100k at d=64.
- **Workloads:** five query-difficulty distributions with matched mean
  noise (eps mean 0.079–0.082) and increasing spread: constant,
  U[0.06,0.10], U[0.03,0.13], U[0.01,0.15], and bimodal {0.02, 0.14}.
- **Dividend:** work saved, measured in distance computations, versus the
  cheapest fixed budget that clears the target on validation.

**Results.** Dividends by spread (left to right: zero → largest) and
target.

| target | seed | const | narrow | medium | wide | bimodal |
|---|---|---|---|---|---|---|
| 0.90 | 701 | +9.3%‡ | +12.2% | +15.8% | +21.9% | **+51.5%** |
| 0.90 | 419 | +13.2%† | +12.0% | +19.7% | +24.3% | +21.0%§ |
| 0.95 | 701 | +8.9%‡ | +10.4% | +10.9% | +21.9% | **+51.9%** |
| 0.95 | 419 | +7.8%† | +17.6%† | +21.7% | +26.9% | +33.1%§ |
| 0.99 | 701 | +9.5% | −0.9%‡ | +6.8%‡ | +14.4%‡ | +28.8% |
| 0.99 | 419 | +10.2%‡ | +15.2% | +10.7%† | +10.4% | +35.9%† |

- † The controller missed the mean-recall contract on test while the
  baseline met it, so the dividend is inflated.
- ‡ Both the controller and the baseline missed the target.
- § The v1 calibration floor applied: the controller overshot, so the
  dividend is understated.
- Every miss is ≤0.4 points of mean recall. That is the val→test slippage
  of empirical bisection, the weakness exp11's conformal calibration
  removes.

**What holds:**
- **Seed 701 (clean, nothing floored)** rises strictly with spread at both
  0.90 and 0.95, from about 9% at zero injected variance to about 52% at
  bimodal. At these two targets, seed 701's baseline budgets were
  identical across all five distributions (ef 179 and 310), so the bimodal
  dividend comes from the controller stopping easy queries early, not from
  a cheaper baseline. On seed 419 the bimodal baseline picked smaller
  budgets at 0.90/0.95 only (149/258); at 0.99 every distribution except
  narrow (768) picked 642.
- **Seed 419's apparent inversions at 0.90 are artifacts the flags
  explain.** The high const cell is contract-inflated (†) and the low
  bimodal cell is floor-understated (§).
- **"About zero at zero variance" is refuted on both seeds.** Constant
  noise still yields +7.8–13.2% at every target. Five of the six constant
  cells carry a flag, and the top value (13.2%) is a † cell whose dividend
  is inflated. But the controller misses in those cells are at most 0.4
  points of recall, which cannot plausibly erase dividends of 8–13%. The
  one fully clean constant cell (seed 701 at 0.99, both methods met)
  still shows +9.5%.
- **Amended hypothesis (post hoc, not yet tested):** the dividend tracks
  *total realized* difficulty variance, of which injected spread is only
  one component; the rest would come from intrinsic per-region
  difficulty such as local density and hubness. exp12 did not measure
  this. Testing it needs a per-query difficulty measure, for example the
  smallest fixed budget each query needs, so realized variance can be
  computed directly.

**What doesn't hold:**
- **The trend at 0.99.** The registered prediction said the dividend
  rises with spread at *every* target. At 0.99 it is not monotone on
  either seed (seed 701: +9.5, −0.9, +6.8, +14.4, +28.8). In that row the
  controller missed its contract in 6 of 10 cells and both methods missed
  in 4 of 10, so calibration and selection noise near the substrate's
  ceiling dominate. Bimodal is still the highest 0.99 cell on both seeds.
- **"Higher target, bigger dividend."** On seed 701, dividends mostly
  *fall* from 0.90 to 0.99 (bimodal 51.5 → 28.8, wide 21.9 → 14.4,
  medium 15.8 → 6.8, narrow 12.2 → −0.9). A higher target does not by
  itself mean a larger dividend.

**Practical reading.** Production traffic that mixes near-duplicate
lookups with exploratory queries looks bimodal. That is where adaptive
termination halves the work.

## 3. Cross-corpus transfer of the controller (exp13): failed, scoped

**Question.** Is per-corpus training necessary? Registered predictions:
- v1: scale-free features transfer better.
- v2 (a): fixing the calibration floor shrinks the transfer failure.
- v2 (b): re-standardizing features on unlabeled target data recovers
  part of the gap to native. This variant is deployable without target
  ground truth.

**Withdrawn.** The v1 synthetic→Reuters magnitudes (−112% to −232%) came
from cells that were all pinned at the calibration floor, the native
controller included. They are a protocol artifact.

**v2 results (seed 617, no floored cells).** Dividends versus the
val-selected fixed baseline:

| direction | target | native | transfer | scale-free | re-standardized |
|---|---|---|---|---|---|
| synthetic → Reuters | 0.90 | +39.2% | −65.1% | +9.7% | −93.1% |
| synthetic → Reuters | 0.95 | +59.1% | −58.7% | −12.8% | −54.6% |
| Reuters → synthetic | 0.90 | +29.2%† | −4.4% | −56.0% | −91.9% |
| Reuters → synthetic | 0.95 | +24.3% | −8.7% | −35.4% | −64.6% |

† The native controller misses the 0.90 mean-recall contract by 0.07
points; the transferred cells all meet it.

In the synthetic→Reuters 0.95 row, the val-selected fixed baseline itself
missed the target on test (recall 0.9477) while every controller variant
met it. A compliant baseline would cost more, so every dividend in that
row is understated.

**Valid seed-523 cells.** Only the synthetic→Reuters half of v1 was pinned
at the floor. The Reuters→synthetic cells were not (tau 0.86–1.06, well
inside the v1 range), so they stand as a second seed for that direction.
The re-standardized variant did not exist in v1.

| direction | target | native | transfer | scale-free |
|---|---|---|---|---|
| Reuters → synthetic | 0.90 | +28.8%† | +2.0% | −30.5% |
| Reuters → synthetic | 0.95 | +24.6% | +3.3% | −23.2% |

† Native misses the contract (0.8983).

**Verdicts:**
- **(a) confirmed.** The plain-transfer failure shrank from −140/−232% to
  −59/−65%. The comparison spans two seeds, so this is indicative rather
  than a controlled measurement.
- **(b) refuted.** Re-standardization was the worst variant in 3 of 4
  cells. In the fourth (synthetic→Reuters at 0.95) it beat plain transfer
  by 4.1 points but stayed 113.7 points below native.
- **Scale-free features: mixed, and seed-dependent.** On seed 617 they
  give the only positive transferred dividend (+9.7%, synthetic→Reuters
  0.90). On seed 523, plain transfer is the positive one (+2.0%, +3.3%)
  and scale-free is strongly negative.
- **Across both seeds, transfer kept little of the native dividend.** In
  all six valid cells, the best transferred variant kept at most a
  quarter of the native controller's dividend, and in 3 of the 6 it fell
  below the plain fixed baseline. The closest it came was seed 523,
  Reuters→synthetic at 0.95: +3.3% against native +24.6%, a 21-point
  gap.

**Conclusion, scoped to what was tested:** across these two corpora, with
this ridge feature set and these three adaptation strategies, the
controller did not transfer usefully. Per-corpus training was needed:
about 600 labeled queries in one instrumented pass. That is cheap but not
optional. Corpus-statistics-conditioned predictors are the obvious next
step.

The native results on the corrected protocol also confirm the Part 3
disclosure. On a Reuters target the native controller saves 39–59% at
0.90/0.95, well above exp10's floor-depressed 3–16%.

## What this adds up to

A calibrated, adaptively terminated search layer with three properties:

1. **Per-query recall guarantees** that held 24/24 across two seeds,
   where the default "pick the cheapest config that passed" approach
   failed 10/24. On a synthetic corpus the guarantee also cost less work.
2. **A tested, partly refuted prediction about when adaptivity pays.**
   On the clean seed the dividend rises strictly with injected difficulty
   spread at 0.90 and 0.95. Bimodal difficulty, which is what mixed
   production traffic looks like, halves the work there. Two parts of the
   registered prediction failed: the dividend does not vanish at zero
   injected spread, and the trend does not hold at 0.99. The "total
   realized variance" explanation is a post-hoc hypothesis that has not
   been tested yet. A target-height effect was never tested, and on
   synthetic data dividends mostly fall as the target rises.
3. **A measured deployment boundary:** for this model class, the best
   transferred controller kept at most a quarter of the native dividend,
   so per-corpus training is required.

Our 2024–26 literature sweep (FRONTIER.md) is not exhaustive. Within it,
the DARTH / LAET / Ada-ef line reports empirical target-hitting rather
than distribution-free guarantees, so the conformal contract appears to
be the new piece.

**Next steps toward external-grade results:**
- A hierarchical substrate, so GloVe-class corpora and the ≥0.999 regime
  become reachable.
- The same harness inside a production graph library, for wall-clock
  confirmation.
- Difficulty-conditional (Mondrian) conformal calibration, to cut the
  Reuters-style conservatism.
