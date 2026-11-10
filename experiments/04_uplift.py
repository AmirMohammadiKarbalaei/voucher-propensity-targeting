"""
Uplift models on Criteo (real randomised data).

Question: if we can only target some people, who should we target?
  - Response model ("who is likely to act if targeted?"): ranks people by P(act | ads).
    This is what our REES46 propensity model does.
  - Uplift model ("whose behaviour do ads change?"): ranks people by
        P(act | ads) - P(act | no ads)
    Only possible because treatment was randomised, so the two groups are alike.
      T-learner: two models, one trained on treated users, one on control users; subtract.
      S-learner: one model with "treatment" as an extra input; predict with it set to 1 and to 0; subtract.

Evaluation on held-out users (Qini curve): rank by each score; for the top k%, compare treated vs
control in that slice -> extra actions caused by ads. Shown as a share of the extra actions you'd
get by targeting everyone. Random targeting is the baseline (a straight diagonal line).

Run for two outcomes: conversion (rare, the business metric) and visit (13x more common).

Dataset caveat: Criteo pooled and sub-sampled the data and anonymised the features (f0-f11), so
the numbers are valid for this dataset only.

    uv run python experiments/04_uplift.py
"""
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "criteo" / "criteo-research-uplift-v2.1.csv.gz"
REPORTS = ROOT / "reports"
FEATURES = [f"f{i}" for i in range(12)]
OUTCOMES = ["conversion", "visit"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=500,
              n_estimators=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=0, verbose=-1)


def fit(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**PARAMS).fit(X, y)


def qini_curve(score: np.ndarray, treated: np.ndarray, y: np.ndarray, steps: int = 100) -> pd.DataFrame:
    """Extra actions caused by ads if we target the top k% by score (counted over the test set)."""
    order = np.argsort(-score, kind="stable")
    t, yy = treated[order], y[order]
    cum_t, cum_c = np.cumsum(t), np.cumsum(1 - t)
    cum_yt, cum_yc = np.cumsum(yy * t), np.cumsum(yy * (1 - t))
    rows = []
    for k in np.linspace(0, 1, steps + 1)[1:]:
        i = int(k * len(order)) - 1
        lift = cum_yt[i] / max(cum_t[i], 1) - cum_yc[i] / max(cum_c[i], 1)   # treated rate - control rate
        rows.append({"share_targeted": k, "extra_actions": lift * (i + 1)})
    return pd.DataFrame(rows)


def scores_for(outcome: str) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(DATA, usecols=FEATURES + ["treatment", outcome],
                     dtype={**{f: "float32" for f in FEATURES}, "treatment": "int8", outcome: "int8"})
    # Stratify on treatment x outcome so both halves keep the same mix of rare outcomes
    train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df.treatment * 2 + df[outcome])
    tr_t, tr_c = train[train.treatment == 1], train[train.treatment == 0]

    model_t, model_c = fit(tr_t[FEATURES], tr_t[outcome]), fit(tr_c[FEATURES], tr_c[outcome])   # T-learner
    p_t, p_c = model_t.predict_proba(test[FEATURES])[:, 1], model_c.predict_proba(test[FEATURES])[:, 1]
    model_s = fit(train[FEATURES + ["treatment"]], train[outcome])                              # S-learner
    s1 = model_s.predict_proba(test[FEATURES].assign(treatment=1))[:, 1]
    s0 = model_s.predict_proba(test[FEATURES].assign(treatment=0))[:, 1]

    return {
        "uplift: S-learner": s1 - s0,
        "uplift: T-learner": p_t - p_c,
        "response model (likely to act with ads)": p_t,
        "random": np.random.default_rng(0).random(len(test)),
    }, test


def main() -> None:
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, outcome in zip(axes, OUTCOMES):
        scores, test = scores_for(outcome)
        treated, y = test["treatment"].to_numpy(), test[outcome].to_numpy()
        curves = {name: qini_curve(s, treated, y) for name, s in scores.items()}
        everyone = curves["random"]["extra_actions"].iloc[-1]          # target all = same for every ranking
        for name, c in curves.items():
            share = c.extra_actions / everyone
            row = {"outcome": outcome, "score": name, "extra_if_target_everyone": everyone}
            for k in [0.1, 0.2, 0.3]:
                row[f"captured_top{int(k * 100)}pct"] = share[np.isclose(c.share_targeted, k)].iloc[0]
            # Qini coefficient: area between the curve and the random diagonal (higher = better ranking)
            row["qini"] = np.trapezoid(share - c.share_targeted, c.share_targeted)
            rows.append(row)
            ax.plot([0, *c.share_targeted * 100], [0, *share * 100], label=name,
                    linestyle="--" if name == "random" else "-")
        ax.set(xlabel="% of users targeted (highest score first)",
               ylabel="% of all extra actions captured", title=f"Outcome: {outcome}")
        ax.legend(fontsize=8)
        print(f"{outcome}: test set {len(test):,} users; targeting everyone causes {everyone:,.0f} extra")

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "criteo_uplift_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.drop(columns="extra_if_target_everyone").to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig.suptitle("Criteo: who should we target? (Qini curves, held-out users)")
    fig.tight_layout()
    fig.savefig(REPORTS / "criteo_uplift_qini.png", dpi=120)


if __name__ == "__main__":
    main()
