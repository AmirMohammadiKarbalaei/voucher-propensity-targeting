# Monitoring runbook: what to do when a weekly check fires

The weekly pipeline (`airflow/dags/ecom_pipeline.py`) runs the checks in `monitoring/checks.py` every Monday:

- **Day 0** (before the voucher send): is the data usable, have the inputs moved, have the scores moved?
- **Day 7** (the next Monday, once last week's answers are in): were the probabilities right, is the ranking still good?

Every check gives one of three results:

| Result | Meaning | Vouchers this week |
|---|---|---|
| **STOP** | The data is broken, so the ranking can't be trusted | **Not sent** until a person has fixed or approved it |
| **ALERT** | Something changed | **Sent as usual**; a person looks this week |
| **OK** | Nothing to do | Sent |

Why ALERT doesn't block the send: in November 2019 the ranking held up under every kind of drift we saw. Precision in the top 50,000 stayed at 24.8–25.1% under all six fixes tested (`reports/fix_options_18nov.csv`). The *levels* (expected buyers, profit) were what went wrong, and those are handled separately. Broken data is different: an empty or doubled input makes the ranking nonsense, so it stops the send.

## The rule

| When | Check | Threshold | Result | What it caught in November |
|---|---|---|---|---|
| Day 0 | Users to score vs **last week** | more than ±30% | STOP | — (first version compared with 4 Nov and falsely stopped 23 Dec: +33% from normal growth) |
| Day 0 | Duplicate users | any | STOP | — |
| Day 0 | A feature's missing share rises | more than 20 points | STOP | (fault-injection test: an emptied feature) |
| Day 0 | A feature becomes constant | it varied before | STOP | (fault-injection test) |
| Day 0 | Invalid scores (missing, below 0, above 1) | any | STOP | — |
| Day 0 | Feature PSI vs the reference week | > 0.25 | ALERT | 18 Nov: cart features 0.27–0.32 |
| Day 0 | Score PSI vs the reference week | > 0.1 | ALERT | 25 Nov: 0.97, 2 Dec: 0.88 (sale flag switched on) |
| Day 0 | Mean score ÷ normal buy rate | outside 0.7–1.3 | ALERT | 18 Nov: 1.46; 25 Nov: 1.86; 2 Dec: 1.81 |
| Day 7 | Expected ÷ actual buyers | outside 0.8–1.25 | ALERT | 11 Nov: 0.58 (sale week); 18 Nov: 1.67; 25 Nov: 2.0; 16 Dec: 0.69 (Christmas) |
| Day 7 | Lift in the top 50,000 vs the median of earlier weeks | more than 25% lower | ALERT: **retrain** | — (18 Nov: 12.3 vs 8.1) |

Thresholds are in one place (`monitoring/checks.py`), so changing one doesn't mean editing the pipeline.

**Why lift and not precision for the retrain check:** precision depends on how many people buy that week. In the sale week it was 34.6%, simply because twice as many people bought. Lift (precision ÷ the week's buy rate) asks "how much better than random", which is fairer from one week to the next. The training week is left out of the comparison because the model has seen it, which flatters it.

## What a person does

### STOP: the send is blocked
1. Read the failing check in the `day0_checks` task log in Airflow.
2. **Users to score / duplicates:** look at the gold row count per cut-off against the last few weeks (steady growth is normal: 3.0M → 4.0M users from 4 Nov to 23 Dec). Usual causes: a partial Databricks run, or a week loaded twice. Rerun `run_databricks_job`.
3. **Missing share / constant feature:** find the upstream column (silver → sessions → gold). Usual causes: a renamed source column or a failed join. Fix it and rerun from the failed task.
4. **Invalid scores:** check the scoring step: the model version and the feature order.
5. If nothing can be fixed today, choose one: send last week's list again, use the cart rule (the old method), or skip the week. Record which, and why.

### ALERT: feature PSI (an input moved)
1. Open the Evidently report for the week (`monitoring/01_data_drift.py` writes the HTML) and look at the drifted features.
2. Decide which kind of change it is:
   - **Logging change** (the same behaviour recorded differently, e.g. carts in November): the feature no longer means what it did in training. Plan to drop or rebuild it. In November, dropping the cart features cost nothing in ranking (25.0% vs 24.8%) and improved calibration.
   - **Real behaviour change** (people really are browsing more): keep the feature and expect the levels to be off. Watch the day-7 check.
3. Write it down in the weekly log.

### ALERT: mean score ÷ normal buy rate (levels suspect)
1. Vouchers go out as usual: the ranking decides who gets them.
2. For this week's money estimates (expected buyers, profit), **anchor** the scores: shift every user's log-odds by one constant so the average equals a normal week's buy rate (`monitoring/04_production_calibration.py`). For a week known to be unusual (Black Friday), anchor to the same week last year instead.
3. Report profit as a **range** (anchored to translated), not one number. On 18 Nov (model without cart features, `reports/production_calibration_18nov.csv`): 80,000 (anchored) to 106,000 (translated) expected buyers; actual 69,838.

### ALERT: expected ÷ actual buyers (day 7)
1. Was last week normal? Check the calendar (sales, holidays) and the actual buy rate against recent weeks.
2. **Normal week:** refit the translator (Platt) on that week's out-of-fold scores. Check it on a held-out week before using it: keep it only if it doesn't make expected ÷ actual worse.
3. **Unusual week** (11 Nov, Singles' Day): don't fit the translator on it, and don't train on it without a flag (`sale_week`). Note it so the next planned retrain handles it.

### ALERT: lift dropped (retrain)
The ranking itself has got worse, and recalibration can't fix that.
1. Retrain on the most recent labelled weeks, with unusual weeks flagged and any feature with a known logging change dropped.
2. Compare the new model with the current one on the latest labelled week, using precision in the top 50,000 with a paired bootstrap CI (`monitoring/03_fix_options.py`).
3. Deploy only if it's at least as good. Register it as a new model version and record why.

## Replay: what the rule would have said in November 2019

`uv run python monitoring/05_weekly_checks.py replay` (model trained on the 4 Nov week; results in `reports/weekly_checks.csv`):

| Week | Day 0 | Day 7 |
|---|---|---|
| 11 Nov (Singles' Day sale) | OK: inputs and scores looked normal | ALERT: expected ÷ actual 0.58 (more buyers than predicted) |
| 18 Nov | ALERT: cart features drifted (PSI up to 0.32); mean score 1.46× the normal buy rate | ALERT: expected ÷ actual 1.67; lift 12.3, no retrain |
| 18 Nov with `n_events_7d` deliberately emptied (fault injection) | **STOP**: missing share +100%, feature constant | — |

What this shows: input checks can't see a change in behaviour such as a sale week (11 Nov passed every day-0 check). That only shows up on day 7. This is why both layers exist.

## Replay 2: Black Friday to Christmas (model in use after the 18 Nov alarm)

`uv run python monitoring/06_later_weeks.py`: the model without cart features and with the sale flag, trained on 4 + 11 Nov and scored every Monday. Results in `reports/later_weeks.csv` and `reports/later_weeks_checks.csv`. Evidently reports for the Black Friday week are in `docs/monitoring/data_drift_2019-11-25.html` and `classification_2019-11-25.html`.

| Week | Actual buy rate | Expected ÷ actual | Precision@50k (cart rule) | Day 0 | Day 7 |
|---|---|---|---|---|---|
| 18 Nov | 2.04% | 1.52 | 25.0% (5.8%) | ALERT: levels high | ALERT |
| 25 Nov (Black Friday, flag on) | 2.17% | 2.00 (1.13 with flag off) | 30.8% (8.3%) | ALERT: scores jumped | ALERT |
| 2 Dec (Cyber Monday, flag on) | 2.03% | 2.08 (1.20 with flag off) | 30.9% (8.2%) | ALERT: scores jumped | ALERT |
| 9 Dec | 2.09% | 1.13 | 33.3% (8.4%) | OK | OK |
| 16 Dec | 3.51% | 0.69 | 43.9% (15.1%) | OK | ALERT: Christmas buying, under-predicted |
| 23 Dec | 3.14% | 0.83 | 41.1% (12.8%) | OK | OK |

What it shows:
- **The ranking held for six weeks**: lift 12–16, 3–5 times the cart rule's precision. The lift check never asked for a retrain.
- **Black Friday was not a sale week for this shop's buyers**: the buy rate barely moved (2.17% vs about 2.0%). The sale flag, learned from Singles' Day (when buying doubled), assumed every sale behaves like that one, and doubled the predictions. With the flag off: 1.13× and slightly better precision. A flag needs evidence for each kind of event, not just a calendar entry. Turn it on only for events that have already been shown to lift buying.
- **Christmas buying (16 Dec) passed every day-0 check**: inputs and scores looked normal, but buying rose to 3.5%. Only the day-7 check caught it. The same lesson as the Singles' Day week: behaviour changes show up when the answers arrive.
- **Fixed in the rule**: the user-count check now compares with last week.
