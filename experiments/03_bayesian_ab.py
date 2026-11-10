"""
The same Criteo A/B test, analysed the Bayesian way (real data).

Instead of "is p < 0.05?", the Bayesian analysis answers the questions a manager actually asks:
  - How likely is it that ads beat no ads?            -> P(treatment > control)
  - If we choose ads and we're wrong, how much do we lose? -> expected loss
  - What's the plausible range for the gain?          -> 95% credible interval

Method: Beta-Binomial (conjugate). Start from a flat prior Beta(1, 1) for each group's conversion
rate; after seeing s buyers out of n users, the belief becomes Beta(1 + s, 1 + n - s). No MCMC is
needed: we draw random samples from the two Beta distributions and compare them.

To show how certainty grows with data, we analyse random subsamples of the real data (10k, 50k,
200k users) as well as all 14M users. The subsamples are real users, not simulated.

    uv run python experiments/03_bayesian_ab.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "criteo" / "criteo-research-uplift-v2.1.csv.gz"
REPORTS = ROOT / "reports"
DRAWS = 200_000
SIZES = [10_000, 50_000, 200_000, None]          # None = all users
LOSS_THRESHOLD_PTS = 0.01                        # "we can live with losing 1 buyer per 10,000 users"
rng = np.random.default_rng(0)


def posterior_draws(buyers: int, users: int) -> np.ndarray:
    """Samples of the conversion rate after a flat Beta(1, 1) prior and the observed data."""
    return rng.beta(1 + buyers, 1 + users - buyers, size=DRAWS)


def analyse(df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    t, c = df[df.treatment == 1].conversion, df[df.treatment == 0].conversion
    rate_t, rate_c = posterior_draws(int(t.sum()), len(t)), posterior_draws(int(c.sum()), len(c))
    diff = (rate_t - rate_c) * 100                                   # percentage points
    return {
        "users": len(df), "buyers_treatment": int(t.sum()), "buyers_control": int(c.sum()),
        "p_treatment_better": (diff > 0).mean(),
        "diff_mean_pts": diff.mean(),
        "cri_low_pts": np.quantile(diff, 0.025), "cri_high_pts": np.quantile(diff, 0.975),
        # Expected loss: on average, how many points we give up if we pick this option and it's the worse one
        "loss_if_choose_ads_pts": np.maximum(-diff, 0).mean(),
        "loss_if_choose_no_ads_pts": np.maximum(diff, 0).mean(),
    }, diff


def main() -> None:
    df = pd.read_csv(DATA, usecols=["treatment", "conversion"], dtype="int8")
    rows, diffs = [], {}
    for size in SIZES:
        sample = df if size is None else df.sample(n=size, random_state=0)
        row, diff = analyse(sample)
        row["decision"] = "ship ads" if row["loss_if_choose_ads_pts"] < LOSS_THRESHOLD_PTS else "keep testing"
        rows.append(row)
        diffs[f"{len(sample):,} users"] = diff

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "criteo_bayesian_results.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nDecision rule: ship if expected loss of choosing ads < {LOSS_THRESHOLD_PTS} pts "
          f"({LOSS_THRESHOLD_PTS * 100:.0f} buyer per 10,000 users)")

    # One panel per sample size (shared x-axis), so the narrow full-data curve doesn't flatten the others
    fig, axes = plt.subplots(len(diffs), 1, figsize=(7, 7), sharex=True)
    for ax, (label, diff), row in zip(axes, diffs.items(), rows):
        per_10k = diff * 100
        counts, edges = np.histogram(per_10k, bins=150)
        colours = np.where(edges[:-1] < 0, "tab:red", "tab:blue")                 # red = "ads are worse"
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=colours, alpha=0.8)
        ax.axvline(0, color="grey", linestyle="--")
        ax.set_yticks([])
        ax.set_title(f"{label}: P(ads better) = {row['p_treatment_better']:.1%}", fontsize=10, loc="left")
    axes[-1].set(xlabel="Extra buyers per 10,000 users because of ads (red = ads worse)", xlim=(-90, 60))
    fig.suptitle("Criteo: what we believe about the ad effect, as data grows")
    fig.tight_layout()
    fig.savefig(REPORTS / "criteo_bayesian_posteriors.png", dpi=120)


if __name__ == "__main__":
    main()
