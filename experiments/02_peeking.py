"""
Why "peeking" at an A/B test inflates false wins (SIMULATED data).

We simulate A/A tests: both groups have exactly the same buying rate (2%), so any "significant"
result is a false positive. Each test runs 20 days; we check the z-test p-value every day.

  - Check once at the end            -> false-positive rate about 5% (as promised)
  - Check daily, stop at first p<0.05 -> much higher
  - Fix: a stricter threshold at every look (constant boundary, found here by simulation),
    or decide the sample size up front and look once.

    uv run python experiments/02_peeking.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
N_SIMS, DAYS, USERS_PER_DAY, RATE = 10_000, 20, 5_000, 0.02  # users per group per day
rng = np.random.default_rng(0)


def z_stats() -> np.ndarray:
    """z-statistic after each day for every simulated A/A test: shape (N_SIMS, DAYS)."""
    a = rng.binomial(USERS_PER_DAY, RATE, size=(N_SIMS, DAYS)).cumsum(axis=1)
    b = rng.binomial(USERS_PER_DAY, RATE, size=(N_SIMS, DAYS)).cumsum(axis=1)
    n = USERS_PER_DAY * np.arange(1, DAYS + 1)
    pooled = (a + b) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * 2 / n)
    return (a / n - b / n) / se


def main() -> None:
    z = np.abs(z_stats())
    crit = stats.norm.ppf(0.975)                                   # 1.96: the usual 5% threshold

    once = (z[:, -1] > crit).mean()
    rows = []
    for looks in [1, 2, 5, 10, 20]:
        idx = np.linspace(DAYS / looks, DAYS, looks).round().astype(int) - 1   # evenly spaced looks
        rows.append({"looks": looks, "false_positive_rate": (z[:, idx] > crit).any(axis=1).mean()})
    table = pd.DataFrame(rows)

    # Fix: one constant threshold for all 20 looks, chosen so the overall false-positive rate is 5%
    max_z = z.max(axis=1)
    boundary = np.quantile(max_z, 0.95)
    fixed = (max_z > boundary).mean()
    bonferroni = stats.norm.ppf(1 - 0.05 / (2 * DAYS))

    table.to_csv(REPORTS / "peeking_false_positives.csv", index=False)
    print(f"{N_SIMS:,} simulated A/A tests (no real difference), {DAYS} days, {USERS_PER_DAY:,} users/group/day\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1%}"))
    print(f"\nLook once at the end: {once:.1%}")
    print(f"Fix: threshold |z| > {boundary:.2f} at every daily look (instead of 1.96) -> {fixed:.1%}")
    print(f"(Bonferroni would use |z| > {bonferroni:.2f}: safe but stricter than needed, "
          f"because the looks share data)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["looks"], table["false_positive_rate"] * 100, marker="o")
    ax.axhline(5, color="grey", linestyle="--", label="promised 5%")
    ax.set(xlabel="Number of times you check the result", ylabel="False wins (%)",
           title="Peeking at an A/A test (simulated)", xticks=table["looks"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "peeking_false_positives.png", dpi=120)


if __name__ == "__main__":
    main()
