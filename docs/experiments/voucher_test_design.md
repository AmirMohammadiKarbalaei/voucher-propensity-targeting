# Design: voucher A/B test for the REES46 shop (SIMULATED)

Written 2026-11-03. **REES46 has no vouchers, so this test was never run.** Populations and baseline
buy rates are real (test week, 18 Nov 2019, calibrated scores). The money figures and voucher effects
are **assumptions**, and the "result" below is a **simulation** used to check that the design and
analysis work. Script: `experiments/05_voucher_test_design.py`.

## Why a test is needed
The model ranks users by how likely they are to buy (24.8% of its top 50,000 buy). That is not the
same as how much a voucher *changes* their buying. Many of the top 50,000 may be "sure things" who
would buy anyway, so a voucher just gives them a discount. Only a randomised test can measure the
extra buyers a voucher causes, and in which group.

## Question
For each candidate group, does sending a voucher cause enough extra purchases to pay for the
discounts? Which group gives the most profit per voucher?

## Unit and randomisation
- Unit: **user**. Assignment: a hash of `user_id` with a fixed salt, mod 2 → voucher / no voucher.
  Repeatable and sticky: a user who appears in both weeks stays in the same arm.
- Randomised **within each group** (stratified), 50/50.

## Groups (strata) and budget
| Group | Who | Users / week | Vouchers / week | Real baseline buy rate |
|---|---|---|---|---|
| A | model top 50,000 | 50,000 | 25,000 | 24.8% |
| B | "on the fence": model ranks 50,001–150,000, random 30,000 | 30,000 | 15,000 | 9.4% |
| C | current cart rule, outside A and B, random 20,000 | 20,000 | 10,000 | 4.0% |
| | **Total** | 100,000 | **50,000** (the existing weekly budget) | |

## Metrics
- **Primary:** bought within 7 days of the voucher (per user, as randomised: intention to treat).
- **Decision metric:** profit per 1,000 vouchers = extra orders × profit per order − $5 on every
  voucher order (including people who would have bought anyway).
- **Guardrails:** revenue per user; share of orders that use a voucher; unsubscribes or complaints.
- **Not allowed:** comparing people who *used* the voucher with the no-voucher group (users choose
  to redeem, so it's biased, like "exposed vs control" in the Criteo test).

## Assumed economics (to be replaced by finance's numbers)
- Order value $286 (mean spend per purchase in the data, real).
- Margin 10% → $28.60 profit per order (**assumed**).
- Voucher: $5 off (**assumed**).
- Break-even lift = baseline × 5 / (28.60 − 5) ≈ 0.21 × baseline. The higher the baseline, the
  bigger the lift needed, because more voucher users would have bought anyway.

| Group | Break-even lift | MDE, 1 week | MDE, 2 weeks |
|---|---|---|---|
| A | +5.2 pts | 1.1 pts | 0.8 pts |
| B | +2.0 pts | 0.9 pts | 0.7 pts |
| C | +0.9 pts | 0.8 pts | 0.6 pts |

MDE = smallest lift the test can reliably detect (80% power, two-sided α = 5%).

## Duration: 2 full weeks, fixed in advance
- With one week, group C's MDE (0.8) is almost its break-even (0.9), so we couldn't tell "pays"
  from "doesn't". Two weeks bring every MDE well below its break-even.
- Two full weeks also cover weekday/weekend patterns.
- **Avoid sale weeks** (e.g. Singles' Day, Black Friday): effects there don't carry over to normal weeks.
- **No early stopping** on the result (peeking). Daily checks only for bugs, split ratio (SRM) and guardrails.

## Analysis plan
1. SRM check per group (chi-square vs 50/50; flag at p < 0.001).
2. Per group: lift in buy rate with 95% CI (two-proportion z-test), and profit per 1,000 vouchers with 95% CI.
3. Users present in both weeks: analyse per user-week with standard errors clustered by user.
4. Decision rule: send vouchers to a group only if the profit CI is above 0. Rank groups by profit
   per voucher; fill the 50,000 budget from the best group down.
5. Then use the test data to train a voucher **uplift model** (as on Criteo) for finer targeting.

## Design checks (simulated)
2,000 simulated two-week tests per group:
- No real effect (A/A): 5.0%, 5.8%, 4.5% significant → false-positive rate as promised (~5%).
- True effect = MDE: 78–79% significant → power as planned (~80%).
- Hash split: 50.15% got a voucher.

## One simulated run (effects ASSUMED: A +1.5, B +3.0, C +2.0 pts)
| Group | Measured lift (95% CI) | Profit per 1,000 vouchers (95% CI) |
|---|---|---|
| A | +1.3 pts (0.7 to 1.8) | **−$884** (−1,023 to −745) |
| B | +3.0 pts (2.5 to 3.5) | **+$187** (53 to 322) |
| C | +1.9 pts (1.5 to 2.3) | **+$239** (128 to 351) |

Reading it: in this **made-up** scenario, vouchers work in every group (all lifts above 0), but in
group A they lose money: most voucher users there would have bought anyway, so the discounts cost
more than the extra orders earn. This shows what the test *can* reveal. It is not evidence about
REES46. The test exists because we don't know which scenario is true.
