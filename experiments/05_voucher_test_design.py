"""
Design a voucher A/B test for the REES46 shop, then check the design by SIMULATION.

REES46 has no vouchers, so no real test can be run here. Everything about voucher effects and money
below is SIMULATED / ASSUMED and labelled as such. The populations and baseline buy rates are real
(test week, 18 Nov, calibrated scores).

Design (see docs/experiments/voucher_test_design.md):
  Three groups (strata), each split 50/50 by coin flip into voucher / no voucher:
    A  model top 50,000                      (likely buyers; may include many "sure things")
    B  model ranks 50,001-150,000 ("on the fence"), random 30,000 of them
    C  current cart rule, not in A or B, random 20,000 of them
  Vouchers per week: 25,000 + 15,000 + 10,000 = 50,000 (the existing budget).
  Primary metric: bought within 7 days. Decision metric: profit per 1,000 vouchers.

What this script does:
  1. Break-even lift per group (ASSUMED money: $286 order, 10% margin, $5 voucher).
  2. Minimum detectable effect (MDE) per group for 1 and 2 weeks -> choose the duration.
  3. Check the design by simulation: A/A tests (target false-positive rate ~5%) and power at the MDE (target ~80%).
  4. One full SIMULATED test with ASSUMED effects, analysed exactly as the real one would be.

    uv run python experiments/05_voucher_test_design.py
"""
import hashlib
import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SCORES = ROOT / "data" / "scores" / "test_2019-11-18.parquet"
GOLD = ROOT / "data" / "gold" / "cutoff_date=2019-11-18"

# ---- ASSUMED economics (not in the data; a real test would get these from finance) ----
ORDER_VALUE = 286.0          # mean spend per purchase in gold (real), used as the order value
MARGIN_RATE = 0.10           # ASSUMED profit margin on an order
VOUCHER_COST = 5.0           # ASSUMED $5 off, paid on every voucher order (incl. people who'd buy anyway)
PROFIT_PER_ORDER = ORDER_VALUE * MARGIN_RATE

# ---- Design ----
GROUPS = {"A: model top 50k": 50_000, "B: on the fence (ranks 50k-150k)": 30_000,
          "C: cart rule (outside A/B)": 20_000}
ALPHA, POWER = 0.05, 0.80
SALT = "voucher-test-2019-11"   # fixed salt -> a user always lands in the same arm
# ---- SIMULATED true voucher effects (absolute, percentage points), purely for the demo ----
ASSUMED_EFFECT = {"A: model top 50k": 0.015, "B: on the fence (ranks 50k-150k)": 0.030,
                  "C: cart rule (outside A/B)": 0.020}
rng = np.random.default_rng(0)


def arm(user_id: int) -> int:
    """Coin flip that is repeatable: hash the user id with a salt; 1 = voucher, 0 = no voucher."""
    h = hashlib.sha256(f"{SALT}:{user_id}".encode()).hexdigest()
    return int(h, 16) % 2


def build_population() -> pd.DataFrame:
    s = pd.read_parquet(SCORES).sort_values("score_calibrated", ascending=False).reset_index(drop=True)
    s["rank"] = np.arange(1, len(s) + 1)
    rule = pd.read_parquet(glob.glob(str(GOLD / "*.parquet")), columns=["user_id", "carted_not_bought_7d"])
    s = s.merge(rule, on="user_id", how="left")
    a = s[s["rank"] <= 50_000].assign(group="A: model top 50k")
    b = s[(s["rank"] > 50_000) & (s["rank"] <= 150_000)].sample(30_000, random_state=0)
    b = b.assign(group="B: on the fence (ranks 50k-150k)")
    c = s[(s["rank"] > 150_000) & (s.carted_not_bought_7d == 1)].sample(20_000, random_state=0)
    c = c.assign(group="C: cart rule (outside A/B)")
    pop = pd.concat([a, b, c], ignore_index=True)
    pop["voucher"] = pop.user_id.map(arm)
    return pop


def mde(p: float, n_per_arm: int) -> float:
    """Smallest absolute lift detectable with 80% power, two-sided alpha 5% (normal approximation)."""
    z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)
    return z * np.sqrt(2 * p * (1 - p) / n_per_arm)


def analyse(rate_t: float, rate_c: float, n_t: int, n_c: int) -> dict:
    """Lift with 95% CI, p-value, and profit per 1,000 vouchers with 95% CI."""
    lift = rate_t - rate_c
    var_t, var_c = rate_t * (1 - rate_t) / n_t, rate_c * (1 - rate_c) / n_c
    se = np.sqrt(var_t + var_c)
    z = stats.norm.ppf(0.975)
    # Profit per voucher sent = extra orders x profit - voucher cost on every voucher order
    #                         = (PROFIT - COST) x rate_t - PROFIT x rate_c
    profit = (PROFIT_PER_ORDER - VOUCHER_COST) * rate_t - PROFIT_PER_ORDER * rate_c
    se_profit = np.sqrt((PROFIT_PER_ORDER - VOUCHER_COST) ** 2 * var_t + PROFIT_PER_ORDER ** 2 * var_c)
    return {"rate_voucher": rate_t, "rate_no_voucher": rate_c, "lift_pts": lift * 100,
            "lift_ci_low": (lift - z * se) * 100, "lift_ci_high": (lift + z * se) * 100,
            "p_value": 2 * stats.norm.sf(abs(lift / se)),
            "profit_per_1000": profit * 1000, "profit_ci_low": (profit - z * se_profit) * 1000,
            "profit_ci_high": (profit + z * se_profit) * 1000}


def main() -> None:
    pop = build_population()
    weeks = 2

    # 1 + 2. Break-even and MDE per group
    plan = []
    for g, d in pop.groupby("group", sort=False):
        p = d.label.mean()                        # real baseline buy rate in this group (test week)
        n_arm = len(d) // 2
        breakeven = p * VOUCHER_COST / (PROFIT_PER_ORDER - VOUCHER_COST)
        plan.append({"group": g, "users_per_week": len(d), "vouchers_per_week": n_arm,
                     "baseline_rate": p, "breakeven_lift_pts": breakeven * 100,
                     "mde_1_week_pts": mde(p, n_arm) * 100, "mde_2_weeks_pts": mde(p, n_arm * 2) * 100})
    plan = pd.DataFrame(plan)
    plan.to_csv(REPORTS / "voucher_test_plan.csv", index=False)
    pd.set_option("display.width", 200)
    print("PLAN (real baseline rates; ASSUMED economics)")
    print(plan.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Split check: {pop.voucher.mean():.2%} of users got a voucher (target 50%)\n")

    # 3. Check the design: A/A (no effect) and power at the MDE, 2,000 simulated 2-week tests per group
    checks = []
    for g, d in pop.groupby("group", sort=False):
        p, n = d.label.mean(), (len(d) // 2) * weeks
        m = mde(p, n)
        for name, effect in [("A/A (no effect)", 0.0), ("effect = MDE", m)]:
            ct = rng.binomial(n, p + effect, 2000) / n
            cc = rng.binomial(n, p, 2000) / n
            se = np.sqrt(ct * (1 - ct) / n + cc * (1 - cc) / n)
            sig = (np.abs(ct - cc) / se > stats.norm.ppf(1 - ALPHA / 2)).mean()
            checks.append({"group": g, "scenario": name, "share_significant": sig})
    checks = pd.DataFrame(checks)
    print("DESIGN CHECK (SIMULATED): A/A should be ~5%, effect = MDE should be ~80%")
    print(checks.to_string(index=False, float_format=lambda v: f"{v:.1%}"), "\n")

    # 4. One full SIMULATED test: baseline outcome from each user's calibrated score (2 weeks),
    #    plus an ASSUMED voucher effect for users in the voucher arm
    rows = []
    for g, d in pop.groupby("group", sort=False):
        d = pd.concat([d] * weeks, ignore_index=True)                    # same users, two weekly cohorts
        p_buy = d.score_calibrated + d.voucher * ASSUMED_EFFECT[g]
        bought = rng.random(len(d)) < p_buy
        t, c = bought[d.voucher == 1], bought[d.voucher == 0]
        rows.append({"group": g, "assumed_true_lift_pts": ASSUMED_EFFECT[g] * 100,
                     **analyse(t.mean(), c.mean(), len(t), len(c))})
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "voucher_test_simulated_results.csv", index=False)
    print("SIMULATED TEST RESULT (effects ASSUMED, not measured)")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(res))
    ax.errorbar(res.profit_per_1000, y, xerr=[res.profit_per_1000 - res.profit_ci_low,
                                              res.profit_ci_high - res.profit_per_1000],
                fmt="o", capsize=4)
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_yticks(y, res.group)
    ax.invert_yaxis()
    ax.set(xlabel="Profit per 1,000 vouchers ($, 95% CI)",
           title="SIMULATED voucher test (assumed effects and economics)")
    fig.tight_layout()
    fig.savefig(REPORTS / "voucher_test_simulated_profit.png", dpi=120)


if __name__ == "__main__":
    main()
