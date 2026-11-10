"""
Analyse a real randomised experiment (Criteo Uplift v2.1), as pre-registered in
docs/experiments/criteo_preregistration.md.

    uv run python experiments/01_criteo_ab.py

Order: SRM check -> sanity check -> primary metric (conversion) -> secondary (visit, Holm) ->
the biased "exposed vs control" comparison, shown only to measure the bias.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "criteo" / "criteo-research-uplift-v2.1.csv.gz"
REPORTS = ROOT / "reports"
DESIGN_TREATMENT_SHARE = 0.85
ALPHA, SRM_ALPHA = 0.05, 0.001
Z = stats.norm.ppf(0.975)


def compare(success_t: int, n_t: int, success_c: int, n_c: int) -> dict:
    """Two-proportion comparison: rates, z-test, chi-square, CIs for absolute and relative lift."""
    p_t, p_c = success_t / n_t, success_c / n_c
    z, p_value = proportions_ztest([success_t, success_c], [n_t, n_c])          # pooled z-test
    chi2, p_chi2, _, _ = stats.chi2_contingency([[success_t, n_t - success_t],
                                                 [success_c, n_c - success_c]], correction=False)
    diff = p_t - p_c
    se_diff = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)            # unpooled, for the CI
    # Relative lift CI via the log of the ratio (delta method), then back to %
    log_ratio = np.log(p_t / p_c)
    se_log = np.sqrt((1 - p_t) / success_t + (1 - p_c) / success_c)
    return {
        "rate_treatment": p_t, "rate_control": p_c,
        "abs_diff_pts": diff * 100, "abs_ci_low_pts": (diff - Z * se_diff) * 100,
        "abs_ci_high_pts": (diff + Z * se_diff) * 100,
        "rel_lift_pct": (np.exp(log_ratio) - 1) * 100,
        "rel_ci_low_pct": (np.exp(log_ratio - Z * se_log) - 1) * 100,
        "rel_ci_high_pct": (np.exp(log_ratio + Z * se_log) - 1) * 100,
        "z": z, "p_value": p_value, "chi2": chi2, "p_chi2": p_chi2,
    }


def main() -> None:
    df = pd.read_csv(DATA, usecols=["treatment", "conversion", "visit", "exposure"], dtype="int8")
    n = len(df)
    t, c = df[df.treatment == 1], df[df.treatment == 0]
    n_t, n_c = len(t), len(c)

    # 1. SRM: are the group sizes what the design promised?
    srm_chi2, srm_p = stats.chisquare([n_t, n_c], [n * DESIGN_TREATMENT_SHARE, n * (1 - DESIGN_TREATMENT_SHARE)])
    print(f"Users: {n:,}  treatment {n_t:,} ({n_t / n:.4%})  control {n_c:,} ({n_c / n:.4%})")
    print(f"SRM vs 85/15: chi2 = {srm_chi2:.2f}, p = {srm_p:.3g} -> {'FLAGGED' if srm_p < SRM_ALPHA else 'ok'}")

    # 2. Sanity: control users should never have seen an ad
    print(f"Exposure rate: treatment {t.exposure.mean():.4%}, control {c.exposure.mean():.4%}")

    # 3. Primary and secondary metrics, as randomised (intention to treat)
    rows = {m: compare(int(t[m].sum()), n_t, int(c[m].sum()), n_c) for m in ["conversion", "visit"]}
    results = pd.DataFrame(rows).T
    results["role"] = ["primary", "secondary"]
    # Holm across the secondary metrics (only one today; the code works for more)
    sec = results["role"] == "secondary"
    results.loc[sec, "p_holm"] = multipletests(results.loc[sec, "p_value"], alpha=ALPHA, method="holm")[1]
    results.loc[~sec, "p_holm"] = results.loc[~sec, "p_value"]

    # 4. What NOT to do: exposed treated users vs control (exposure isn't randomised)
    exposed = t[t.exposure == 1]
    biased = compare(int(exposed.conversion.sum()), len(exposed), int(c.conversion.sum()), n_c)
    results.loc["conversion: exposed vs control (biased, for illustration)"] = {**biased, "role": "not valid"}

    results.to_csv(REPORTS / "criteo_ab_results.csv")
    pd.set_option("display.width", 220)
    cols = ["role", "rate_treatment", "rate_control", "abs_diff_pts", "abs_ci_low_pts", "abs_ci_high_pts",
            "rel_lift_pct", "rel_ci_low_pct", "rel_ci_high_pct", "z", "p_value", "p_chi2", "p_holm"]
    print(results[cols].to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"\nShare of treated users actually exposed: {len(exposed) / n_t:.2%}")


if __name__ == "__main__":
    main()
