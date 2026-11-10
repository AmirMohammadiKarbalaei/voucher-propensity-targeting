"""
Run the chosen model week by week through Black Friday and December.

Model F (the fix chosen in 03_fix_options.py): no cart features, trained on 4 + 11 Nov with a sale-week flag.
Reference = 4 Nov (the normal training week). Every later week is scored on its Monday and checked
with the weekly rule (monitoring/checks.py) on day 0, then again on day 7 once its answers are in.

The sale flag is known in advance from the calendar, so it is set when scoring:
  25 Nov week: Black Friday (29 Nov) -> sale_week = 1
  2 Dec week:  Cyber Monday (2 Dec)  -> sale_week = 1
Both weeks are also scored with sale_week = 0, to see what the flag changes.

Outputs:
  reports/later_weeks.csv, reports/later_weeks.png, reports/later_weeks_checks.csv
  docs/monitoring/data_drift_2019-11-25.html            (Evidently: inputs, Black Friday week vs 4 Nov)
  docs/monitoring/classification_2019-11-25.html        (Evidently: performance, Black Friday week vs 18 Nov)

    uv run python monitoring/06_later_weeks.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modelling"))
sys.path.insert(0, str(ROOT / "monitoring"))
from checks import day0_checks, day7_checks, psi, verdict  # noqa: E402
from common import BUDGET, LABEL, REPORTS, TRAIN, VALID, feature_columns, load_gold, split, top_k_metrics  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DOCS = ROOT / "docs" / "monitoring"
CART_FEATURES = ["n_carts_30d", "n_carts_7d", "days_since_last_cart", "carts_per_view",
                 "carted_not_bought_7d", "n_cart_visits_7d"]
WEEKS = ["2019-11-18", "2019-11-25", "2019-12-02", "2019-12-09", "2019-12-16", "2019-12-23"]
SALE_WEEKS = {"2019-11-25": "Black Friday", "2019-12-02": "Cyber Monday"}
SAMPLE = 100_000


def anchor(p: np.ndarray, target: float) -> np.ndarray:
    """Shift every log-odds by one constant so the mean score equals target (ranking unchanged)."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(z + mid)))).mean() < target else (lo, mid)
    return 1 / (1 + np.exp(-(z + lo)))


def main() -> None:
    best = json.loads((REPORTS / "best_params.json").read_text())["lightgbm"]
    gold = load_gold()
    feats = [c for c in feature_columns(gold) if c not in CART_FEATURES]
    cols = feats + ["sale_week"]
    train = pd.concat([split(gold, TRAIN).assign(sale_week=0), split(gold, VALID).assign(sale_week=1)],
                      ignore_index=True)
    model = lgb.LGBMClassifier(n_estimators=best["n_trees"], subsample_freq=1, random_state=0, verbose=-1,
                               **best["params"]).fit(train[cols].to_numpy(np.float32), train[LABEL])

    def score(df: pd.DataFrame, flag: int) -> np.ndarray:
        return model.predict_proba(df.assign(sale_week=flag)[cols].to_numpy(np.float32))[:, 1]

    ref = split(gold, TRAIN)
    ref_scores, normal_rate = score(ref, 0), ref[LABEL].mean()

    rows, log, lifts, weeks = [], [], [], {}
    for w in WEEKS:
        df = split(gold, w)
        y = df[LABEL].to_numpy()
        flag = int(w in SALE_WEEKS)
        p = score(df, flag)
        weeks[w] = (df, p)
        m = top_k_metrics(y, p)
        rule = df["carted_not_bought_7d"] * (100 - df["days_since_last_cart"].fillna(30))   # current cart rule
        last = split(gold, WEEKS[WEEKS.index(w) - 1]) if w != WEEKS[0] else split(gold, VALID)
        d0 = day0_checks(ref, df, feats, ref_scores, p, normal_rate, last_week_users=len(last))
        d7 = day7_checks(p, y, lifts, BUDGET)
        lifts.append(m["lift_at_50k"])
        log += [{"week": w, "run": run, **r} for run, rs in [("day 0", d0), ("day 7", d7)] for r in rs]
        top_psi = sorted(((f, psi(ref[f], df[f])) for f in feats), key=lambda t: -t[1])[0]
        row = {
            "week": w, "event": SALE_WEEKS.get(w, ""), "sale_flag": flag, "users": len(df),
            "actual_rate": y.mean(), "mean_score": p.mean(), "score_psi": psi(pd.Series(ref_scores), pd.Series(p)),
            "top_feature_psi": f"{top_psi[0]} {top_psi[1]:.2f}",
            "day0": verdict(d0), "expected_over_actual": p.sum() / y.sum(),
            "anchored_over_actual": anchor(p, normal_rate).sum() / y.sum(),
            "precision_at_50k": m["precision_at_50k"], "lift_at_50k": m["lift_at_50k"], "pr_auc": m["pr_auc"],
            "cart_rule_precision": top_k_metrics(y, rule.to_numpy(float))["precision_at_50k"],
            "day7": verdict(d7),
        }
        if flag:                                       # what the sale flag changed
            p0 = score(df, 0)
            row["expected_over_actual_flag0"] = p0.sum() / y.sum()
            row["precision_at_50k_flag0"] = top_k_metrics(y, p0)["precision_at_50k"]
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "later_weeks.csv", index=False)
    pd.DataFrame(log).to_csv(REPORTS / "later_weeks_checks.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.T.to_string(header=False, float_format=lambda v: f"{v:.4g}"))
    for w in WEEKS:
        alerts = [r for r in log if r["week"] == w and r["level"] != "OK"]
        print(f"\n{w}: " + ("; ".join(f"[{r['run']}] {r['check']} = {r['value']}" for r in alerts) or "all OK"))

    # Evidently reports for the Black Friday week
    bf = "2019-11-25"
    snap = Report([DataDriftPreset()]).run(current_data=weeks[bf][0][feats].sample(SAMPLE, random_state=0),
                                           reference_data=ref[feats].sample(SAMPLE, random_state=0))
    snap.save_html(str(DOCS / f"data_drift_{bf}.html"))

    def perf(w: str) -> Dataset:
        df, p = weeks[w]
        frame = pd.DataFrame({"target": df[LABEL].to_numpy(), "prediction": p}).sample(SAMPLE, random_state=0)
        return Dataset.from_pandas(frame, data_definition=DataDefinition(
            classification=[BinaryClassification(target="target", prediction_probas="prediction")]))
    cutoff = np.sort(weeks["2019-11-18"][1])[-BUDGET]       # score of the 50,000th user in the first week
    snap = Report([ClassificationPreset(probas_threshold=float(cutoff))]).run(
        current_data=perf(bf), reference_data=perf("2019-11-18"))
    snap.save_html(str(DOCS / f"classification_{bf}.html"))

    # Picture
    labels = [f"{pd.Timestamp(w):%d %b}" + (f"\n{SALE_WEEKS[w]}" if w in SALE_WEEKS else "") for w in WEEKS]
    x = np.arange(len(WEEKS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(x, table.actual_rate * 100, "o-", color="black", label="actual buy rate (day 7)")
    axes[0].plot(x, table.mean_score * 100, "o--", color="tab:blue", label="model's mean score (day 0)")
    axes[0].axhline(normal_rate * 100, color="grey", lw=1, ls=":", label=f"normal week (4 Nov) {normal_rate:.2%}")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="%", title="Predicted vs actual buying")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, table.precision_at_50k * 100, "o-", color="tab:green", label="model (top 50,000)")
    axes[1].plot(x, table.cart_rule_precision * 100, "o-", color="tab:orange", label="current rule (50,000)")
    axes[1].plot(x, table.actual_rate * 100, "o:", color="grey", label="random (= buy rate)")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="% who bought within 7 days",
                title="Precision of the 50,000 selected users")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Final model scored weekly, 18 Nov to 23 Dec 2019")
    fig.tight_layout()
    fig.savefig(REPORTS / "later_weeks.png", dpi=120)


if __name__ == "__main__":
    main()
