"""
Two simple baselines and a logistic regression.

- B0 random: 50,000 users at random (the floor).
- B1 current rule: carted but didn't buy in the last 7 days, most recent cart first, capped at 50,000.
- Logistic regression (statsmodels): a transparent model whose coefficients we can read as odds ratios.

Trained on the 4 Nov cutoff, scored on the 11 Nov cutoff (validation). The 18 Nov test week is not touched here.
"""
import mlflow
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import BUDGET, LABEL, MLFLOW_URI, REPORTS, TRAIN, VALID, load_gold, split, top_k_metrics

# Logistic regression treats every number as a straight-line effect, so it needs:
# - no missing values (fill + an "is it present" flag),
# - counts and money on a log scale (otherwise a handful of huge users dominate),
# - no near-duplicate columns, so each coefficient means something on its own. Dropped after a VIF check:
#   views (~ sessions x events per session, VIF 316), spend (~ purchases), and the has_cart / has_purchased
#   flags (already implied by carts / purchases > 0).
CONTINUOUS = [
    "days_since_last_event", "days_since_last_cart", "days_since_last_purchase",
    "log_n_events_7d", "log_n_events_1d", "active_days_30d", "log_n_sessions_30d",
    "log_n_carts_30d", "log_n_purchases_30d",
    "log_avg_price_viewed", "log_n_categories_30d", "carts_per_view",
    "log_events_per_session", "log_longest_visit_min",
]
BINARY = ["carted_not_bought_7d"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["days_since_last_event"] = df["days_since_last_event"]
    x["days_since_last_cart"] = df["days_since_last_cart"].fillna(30)          # never carted = window length
    x["days_since_last_purchase"] = df["days_since_last_purchase"].fillna(30)  # never bought = window length
    for c in ["n_events_7d", "n_events_1d", "n_sessions_30d", "n_views_30d", "n_carts_30d",
              "n_purchases_30d", "avg_price_viewed", "n_categories_30d", "events_per_session", "longest_visit_min"]:
        x[f"log_{c}"] = np.log1p(df[c].fillna(0))
    x["active_days_30d"] = df["active_days_30d"]
    x["carts_per_view"] = df["carts_per_view"].clip(0, 1)
    x["carted_not_bought_7d"] = df["carted_not_bought_7d"]
    return x[CONTINUOUS + BINARY]


def main() -> None:
    gold = load_gold()
    train, valid = split(gold, TRAIN), split(gold, VALID)
    y_valid = valid[LABEL].to_numpy()
    print(f"train {TRAIN}: {len(train):,} users, rate {train[LABEL].mean():.2%}")
    print(f"valid {VALID}: {len(valid):,} users, rate {y_valid.mean():.2%}")

    results = {}

    # B0: random order
    results["B0 random"] = top_k_metrics(y_valid, np.random.default_rng(1).random(len(valid)))

    # B1: the current business rule, most recent cart first, capped by the budget
    b1_score = valid["carted_not_bought_7d"] * (100 - valid["days_since_last_cart"].fillna(30))
    results["B1 current rule"] = top_k_metrics(y_valid, b1_score.to_numpy())
    print(f"B1 eligible users this week: {int(valid['carted_not_bought_7d'].sum()):,} (budget {BUDGET:,})")

    # Logistic regression: standardise continuous features with TRAIN statistics only (no peeking at validation)
    x_train, x_valid = prepare(train), prepare(valid)
    mean, std = x_train[CONTINUOUS].mean(), x_train[CONTINUOUS].std()
    x_train[CONTINUOUS] = (x_train[CONTINUOUS] - mean) / std
    x_valid[CONTINUOUS] = (x_valid[CONTINUOUS] - mean) / std

    # statsmodels gives standard errors and confidence intervals; 500k users is plenty for stable estimates
    sample = x_train.sample(500_000, random_state=0)
    logit = sm.Logit(train.loc[sample.index, LABEL], sm.add_constant(sample)).fit(disp=False, maxiter=200)
    lr_score = logit.predict(sm.add_constant(x_valid)).to_numpy()
    results["Logistic regression"] = top_k_metrics(y_valid, lr_score)

    # Odds ratios: continuous = per 1 standard deviation increase; binary = yes vs no
    ci = logit.conf_int()
    odds = pd.DataFrame({
        "odds_ratio": np.exp(logit.params),
        "ci_low": np.exp(ci[0]),
        "ci_high": np.exp(ci[1]),
        "p_value": logit.pvalues,
    }).drop(index="const").sort_values("odds_ratio", ascending=False)
    odds["per"] = ["1 SD" if f in CONTINUOUS else "yes vs no" for f in odds.index]

    metrics = pd.DataFrame(results).T
    REPORTS.mkdir(exist_ok=True)
    metrics.to_csv(REPORTS / "valid_metrics_baselines.csv")
    odds.to_csv(REPORTS / "odds_ratios.csv")
    std.rename("train_sd").to_csv(REPORTS / "logistic_feature_sd.csv")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("propensity")
    for name, m in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"train_cutoff": TRAIN, "eval_cutoff": VALID, "budget": BUDGET})
            mlflow.log_metrics(m)
            if name == "Logistic regression":
                mlflow.log_artifact(str(REPORTS / "odds_ratios.csv"))

    pd.set_option("display.width", 200)
    print("\nValidation (11 Nov cutoff):")
    print(metrics.round(4).to_string())
    print("\nOdds ratios (logistic regression, trained on 4 Nov):")
    print(odds.round(3).to_string())


if __name__ == "__main__":
    main()
