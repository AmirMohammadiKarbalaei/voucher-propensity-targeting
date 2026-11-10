"""
Calibrate the way production would, on the morning of 18 Nov (no 18 Nov answers yet).

Model F: no cart features, trained on 4 + 11 Nov with a sale-week flag (0 when scoring 18 Nov).
The calibrator (Platt) must be fitted on a week whose answers are already known, using predictions the
model did not train on. So: 5-fold out-of-fold predictions grouped by user on the training rows, then
fit Platt on the rows of the chosen calibration week only.

Compared on 18 Nov (answers checked afterwards):
  - no calibration
  - Platt on out-of-fold predictions for 11 Nov (the latest labelled week, but a sale week)
  - Platt on out-of-fold predictions for 4 Nov (the latest NORMAL labelled week)  <- the rule

    uv run python monitoring/04_production_calibration.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modelling"))
from common import BUDGET, LABEL, REPORTS, TEST, TRAIN, VALID, feature_columns, load_gold, split  # noqa: E402

CART_FEATURES = ["n_carts_30d", "n_carts_7d", "days_since_last_cart", "carts_per_view",
                 "carted_not_bought_7d", "n_cart_visits_7d"]


def model(best: dict) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(n_estimators=best["n_trees"], subsample_freq=1, random_state=0, verbose=-1,
                              **best["params"])


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def main() -> None:
    best = json.loads((REPORTS / "best_params.json").read_text())["lightgbm"]
    gold = load_gold()
    cols = [c for c in feature_columns(gold) if c not in CART_FEATURES] + ["sale_week"]
    train = pd.concat([split(gold, TRAIN).assign(sale_week=0), split(gold, VALID).assign(sale_week=1)],
                      ignore_index=True)
    test = split(gold, TEST).assign(sale_week=0)
    X, y = train[cols].to_numpy(np.float32), train[LABEL].to_numpy()

    # Out-of-fold predictions; grouped by user so a user's 4 Nov and 11 Nov rows are in the same fold
    oof = np.zeros(len(train))
    for fit_idx, pred_idx in GroupKFold(n_splits=5).split(X, y, groups=train["user_id"]):
        oof[pred_idx] = model(best).fit(X[fit_idx], y[fit_idx]).predict_proba(X[pred_idx])[:, 1]

    final = model(best).fit(X, y)                            # the model actually used on 18 Nov
    p_test = final.predict_proba(test[cols].to_numpy(np.float32))[:, 1]
    y_test = test[LABEL].to_numpy()

    results = {"no calibration": p_test}
    for name, week in [("Platt on 11 Nov (sale week)", VALID), ("Platt on 4 Nov (latest normal week)", TRAIN)]:
        rows = (train["cutoff_date"] == week).to_numpy()
        platt = LogisticRegression(C=1e6).fit(logit(oof[rows]), y[rows])
        results[name] = platt.predict_proba(logit(p_test))[:, 1]
        if week == TRAIN:   # check: is the calibrator already close to "no change" on the week it's fitted on?
            print(f"OOF on 4 Nov: mean prediction {oof[rows].mean():.4f} vs actual {y[rows].mean():.4f}; "
                  f"Platt a={platt.coef_[0][0]:.3f}, b={platt.intercept_[0]:.3f}")

    # Day-0 fallback: base-rate anchoring. Shift every user's log-odds by the same amount so the
    # week's total matches the buy rate of the latest normal week. Keeps the ranking exactly.
    target = y[(train["cutoff_date"] == TRAIN).to_numpy()].mean()
    lo, hi = -5.0, 5.0
    for _ in range(60):                                       # bisection for the shift
        mid = (lo + hi) / 2
        mean_p = (1 / (1 + np.exp(-(logit(p_test).ravel() + mid)))).mean()
        lo, hi = (mid, hi) if mean_p < target else (lo, mid)
    results["anchor to 4 Nov buy rate (2.33%)"] = 1 / (1 + np.exp(-(logit(p_test).ravel() + lo)))
    print(f"Anchoring shift: {lo:.3f} on the log-odds scale; 18 Nov input means vs 4 Nov:")
    w4 = split(gold, TRAIN)
    for c in ["n_events_7d", "n_events_1d", "n_views_30d", "n_purchases_7d"]:
        print(f"  {c}: {w4[c].mean():.2f} -> {test[c].mean():.2f}")
    print(f"  buy rate: {w4[LABEL].mean():.4f} -> {y_test.mean():.4f}")

    out = []
    for name, p in results.items():
        top = np.argsort(-p, kind="stable")[:BUDGET]
        out.append({"calibration": name, "expected_buyers": p.sum(), "actual_buyers": int(y_test.sum()),
                    "expected_over_actual": p.sum() / y_test.sum(),
                    "top50k_expected": p[top].sum(), "top50k_actual": int(y_test[top].sum())})
    table = pd.DataFrame(out)
    table.to_csv(REPORTS / "production_calibration_18nov.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
