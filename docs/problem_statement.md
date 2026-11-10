# Problem statement

> **Scenario note:** the stakeholder and retailer are fictional. The event data is real (REES46 public e-commerce logs, 2019). There is no voucher or email data in REES46, so any voucher effect in this project is either measured on a different real experiment (Criteo) or simulated. Simulated results are always labelled as simulated.

## 1. Business context
- **Stakeholder:** Head of CRM at an online multi-category retailer.
- **Decision:** every Monday, choose which users receive a 10%-off voucher, valid for 7 days.
- **Constraint:** at most 50,000 vouchers per week.
- **Current practice:** vouchers go to users who added to basket in the last 7 days and did not buy.
- **Problem raised:** voucher cost is high, and many vouchers go to users who would have bought anyway.
- **Goal:** spend the same 50,000 vouchers on users where they generate more purchases, and show the effect to finance within a quarter.

## 2. Prediction task
| Item | Choice | Reason |
|---|---|---|
| Unit | One row per user per weekly cutoff | The decision is made per user, once a week |
| Cutoff | Monday 00:00 UTC | Matches the Monday send |
| Eligible population | Users with ≥1 event in the 30 days before the cutoff | Dormant users are hard to reach and add noise; this keeps the base rate meaningful. Wider than the current rule, so the model can find buyers the rule misses |
| Observation window | 30 days before the cutoff (with 1-day and 7-day sub-windows) | Long enough for repeat behaviour; short enough to give several cutoffs from limited months of data |
| Horizon | 7 days from the cutoff | Equals voucher validity and the campaign cycle, so horizons don't overlap |
| Label | 1 if the user has ≥1 `purchase` event in [cutoff, cutoff + 7 days), else 0 | The voucher applies to any purchase. A revenue-weighted label is a possible later extension |
| Repeated users | Allowed across cutoffs | More training rows; handled by splitting on time, not at random |
| Split | Train on earlier cutoffs, validate on the next, test on the latest | Mirrors production: always predict the future from the past |

**Model output:** a score per eligible user. The top 50,000 scores receive vouchers.

## 3. Leakage risks
1. **Events after the cutoff in features.** Every feature is computed with `event_time < cutoff`. Checked with a unit test on a hand-built example.
2. **Sessions spanning the cutoff.** A `user_session` can start before the cutoff and end after it. Session features are cut at the cutoff timestamp, not computed per whole session.
3. **Whole-dataset aggregates.** Product or category popularity, average price and similar statistics are computed only from data before each cutoff, not from all months at once.
4. **Random train/test split.** The same user at a later cutoff would let the model "see the future". A time-based split prevents this.
5. **Target or category encodings** fitted on the full dataset. These are fitted on training cutoffs only.
6. **Duplicate events.** Exact duplicate rows are removed before both features and labels are built, so one purchase cannot appear on both sides of the cutoff.

## 4. Metrics
| Level | Metric | Reason |
|---|---|---|
| Model (selection) | PR-AUC | Purchases are rare. ROC-AUC flatters models on imbalanced data |
| Model (decision) | Precision and lift in the top 50,000 | The budget fixes how many users we act on, so only ranking at the top matters |
| Model (supporting) | Calibration (reliability curve, Brier score) | Needed if scores are later turned into expected cost and return |
| Business (primary) | Incremental purchases per 1,000 vouchers, versus a randomised no-voucher holdout | Finance cares about extra purchases caused by vouchers, not predicted ones |
| Business (secondary) | Incremental revenue minus voucher cost | Converts the effect into money |
| Guardrails | Average order value; voucher cost as a share of revenue | The voucher should not just shift full-price orders to discounted ones |

## 5. Baselines
The model is judged against all of these on the same population, cutoffs and metrics:
- **B0, random:** 50,000 eligible users at random. A lower bound.
- **B1, current rule:** users with ≥1 `cart` and no `purchase` in the last 7 days, capped at 50,000 by most recent cart activity.
- **B2, recent-buyer rule:** B1 restricted to users who bought in the last 30 days. Expected to have high precision, but probably many users who would have bought anyway (see section 6).

If the model doesn't beat B1, the recommendation is to keep B1. It is cheaper and easier to explain.

## 6. Assumptions and limitations
- **Propensity is not uplift.** A propensity model ranks users by how likely they are to buy, not by how much the voucher changes that. The highest scorers include many users who would buy anyway, which is the stakeholder's concern. A propensity model on its own therefore improves targeting accuracy but does not prove a lower voucher cost per extra purchase.
- **How this is addressed:**
  - (a) Any rollout is tested against a randomised holdout of eligible users who receive no voucher.
  - (b) Uplift modelling is demonstrated on the Criteo dataset, which has real randomised treatment.
  - (c) The REES46 targeting experiment is designed properly but its outcomes are **simulated**, because REES46 contains no voucher treatment.
- **Data limitations:** one retailer, 2019, with electronics making up a large share. Possible bot traffic. No customer demographics, no email engagement and no unsubscribe data. User IDs may not join anonymous and logged-in activity.
- **Seasonality:** Black Friday (29 Nov 2019) falls inside the data. That makes it a good monitoring test, but it also means test-period results may not generalise.
- **Assumptions:** voucher valid for 7 days on any product; one voucher per user per week; delivery and open rates not modelled.
