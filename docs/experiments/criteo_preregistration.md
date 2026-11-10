# Pre-registration: Criteo advertising experiment (real data)

Written 2026-11-01, **before** looking at any outcome in the data. Only the column names were read.

## Data
Criteo Uplift v2.1 (13,979,592 users; CC BY-NC-SA 4.0, portfolio use only;
https://huggingface.co/datasets/criteo/criteo-uplift). A randomised trial: a random part of the
population was prevented from being targeted by advertising (control); the rest could be targeted
(treatment). Documented design: **85% treatment, 15% control**.

Columns used: `treatment` (randomised assignment), `conversion`, `visit`, `exposure` (whether a
treated user actually saw an ad).

**Caveat stated in the dataset documentation:** the data pools several tests and was sub-sampled
unevenly "so that the original incrementality level cannot be deduced". So the lift measured here is
valid for this dataset, but is not Criteo's real-world advertising effect.

## Question
Does being eligible for advertising (treatment) increase conversions?

## Unit and comparison
- Unit: user. Groups compared **as randomised** (intention to treat): all treatment vs all control,
  whether or not a treated user actually saw an ad.
- Not allowed as the main analysis: "exposed vs control". Exposure happened after randomisation and
  depends on the user (active users see more ads), so that comparison is biased. It is shown once,
  labelled, to show the size of the bias.

## Metrics
- **Primary: conversion rate** (the business outcome). It is rare, but 14M users give enough power.
- **Secondary: visit rate** (an earlier step in the funnel; easier to move, less valuable).
- **Sanity check: exposure rate in control** should be 0 (control users could not be targeted).
- Not used: conversion among visitors. Visiting is itself affected by treatment, so the visitors in
  each group are no longer comparable (post-treatment selection).

## Tests and decision rule
- **Sample-ratio mismatch (SRM) first:** chi-square goodness-of-fit of group sizes against 85/15.
  Flag if p < 0.001 (the usual strict threshold for SRM, because it's checked on every test). If
  flagged, the results are not trusted until explained.
- **Primary:** two-sided two-proportion z-test, α = 0.05; the chi-square test on the 2×2 table as a
  cross-check (same answer for two groups). 95% confidence intervals for the absolute difference
  (percentage points) and the relative lift (%).
- **Secondary:** same tests; Holm correction across the secondary metrics, so extra metrics don't
  raise the chance of a false win.
- **Practical significance:** report the absolute effect, not just the p-value. With 14M users even
  a tiny, worthless effect can be "significant".
- **Recommendation rule:** ship (keep advertising) if the primary metric's 95% CI is above 0;
  otherwise not proven.

## Predictions (written before the run)
- Treatment share close to 85%; SRM not flagged.
- Conversion and visits both higher in treatment; the relative lift on visits larger than on
  conversion is plausible, but conversion is the deciding metric.
- Exposed vs control will show a much bigger "effect" than the randomised comparison.

---

## Results (run 2026-11-01, `experiments/01_criteo_ab.py`, table in `reports/criteo_ab_results.csv`)
| Check / metric | Treatment | Control | Difference (95% CI) | Relative lift (95% CI) | p |
|---|---|---|---|---|---|
| Group sizes | 11,882,655 (85.0000%) | 2,096,937 (15.0000%) | SRM p = 0.999 | | ok |
| Exposure (sanity) | 3.60% | 0.00% | | | ok |
| **Conversion (primary)** | 0.309% | 0.194% | **+0.115 pts (0.108 to 0.122)** | **+59% (54% to 65%)** | < 1e-170 |
| Visit (secondary, Holm) | 4.85% | 3.82% | +1.03 pts (1.01 to 1.06) | +27% (26% to 28%) | < 1e-300 |
| *Exposed vs control (not valid)* | *5.38%* | *0.194%* | *+5.18 pts* | *+2,676%* | |

**Notes**
- The split is exactly 85.0000 / 15.0000. Real experiments are almost never this exact; the
  publisher sub-sampled the data, which likely forced the ratio. So here the SRM check can't detect
  anything, and that is itself worth saying.
- Only 3.6% of treated users actually saw an ad, so the as-randomised effect is diluted. Dividing by
  the exposure share (0.115 / 0.036 ≈ 3.2 pts) estimates the effect on those who were exposed, under
  the assumption that being eligible without seeing an ad does nothing. That is very different from
  the naive +5.18 pts: about 40% of the naive "effect" is selection (exposed users are more active).
- Predictions: SRM not flagged ✓; both metrics up ✓; relative lift larger on visits ✗ (conversion
  +59% vs visits +27%); exposed vs control far bigger ✓.

## Recommendation
Advertising eligibility increases conversion: +0.115 percentage points (95% CI 0.108 to 0.122),
about 1.15 extra conversions per 1,000 users targeted. The interval is well above zero, so by the
pre-registered rule: **keep advertising**. Whether it pays depends on the cost per 1,000 users
targeted versus the value of 1.15 conversions, which this dataset can't give (sub-sampled, no costs).
Never report the exposed-vs-control figure: it overstates the effect by mixing in who chooses to be active.
