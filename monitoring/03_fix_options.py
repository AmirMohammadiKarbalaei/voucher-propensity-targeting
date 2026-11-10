"""
The alarm went off on 18 Nov. Which fix gives the best voucher list?

Options, all available on the morning of 18 Nov (labels for 4 Nov and 11 Nov weeks are known):
  A  keep the model trained on 4 Nov
  B  retrain on 4 + 11 Nov (the model we deployed)
  C  retrain on 4 + 11 Nov with a "sale week" flag (1 for 11 Nov, 0 otherwise; 0 when scoring 18 Nov)
  D  drop the cart features (logging changed), train on 4 Nov
  E  drop the cart features, train on 4 + 11 Nov
  F  both fixes: no cart features, 4 + 11 Nov with the sale-week flag

Judged on the 18 Nov week (test, never used for choices): precision@50k with a paired bootstrap CI
of the difference vs B, plus expected vs actual buyers (calibration, before any recalibration).

    uv run python monitoring/03_fix_options.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modelling"))
from common import BUDGET, LABEL, REPORTS, TEST, TRAIN, VALID, feature_columns, load_gold, split  # noqa: E402

CART_FEATURES = ["n_carts_30d", "n_carts_7d", "days_since_last_cart", "carts_per_view",
                 "carted_not_bought_7d", "n_cart_visits_7d"]
N_BOOT = 200


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], best: dict) -> np.ndarray:
    model = lgb.LGBMClassifier(n_estimators=best["n_trees"], subsample_freq=1, random_state=0, verbose=-1,
                               **best["params"])
    model.fit(train[cols].to_numpy(np.float32), train[LABEL])
    return model.predict_proba(test[cols].to_numpy(np.float32))[:, 1]


def boot_precisions(order: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Precision@50k under Poisson bootstrap weights: walk down the ranking until 50,000 'users' (weights)."""
    w = weights[:, order]                                    # (N_BOOT, users) in ranked order
    cum = np.cumsum(w, axis=1)
    cut = (cum < BUDGET).sum(axis=1)                          # index where the budget runs out
    buyers = np.cumsum(w * y[order], axis=1)
    return buyers[np.arange(len(cut)), cut] / cum[np.arange(len(cut)), cut]


def main() -> None:
    best = json.loads((REPORTS / "best_params.json").read_text())["lightgbm"]
    gold = load_gold()
    cols = feature_columns(gold)
    no_cart = [c for c in cols if c not in CART_FEATURES]
    w4, w11, test = split(gold, TRAIN), split(gold, VALID), split(gold, TEST)
    both = pd.concat([w4.assign(sale_week=0), w11.assign(sale_week=1)], ignore_index=True)
    y = test[LABEL].to_numpy()

    scores = {
        "A: 4 Nov model as is": fit_predict(w4, test, cols, best),
        "B: retrain 4+11 Nov (deployed)": fit_predict(both, test, cols, best),
        "C: retrain 4+11 Nov + sale flag": fit_predict(both, test.assign(sale_week=0), cols + ["sale_week"], best),
        "D: no cart features, 4 Nov": fit_predict(w4, test, no_cart, best),
        "E: no cart features, 4+11 Nov": fit_predict(both, test, no_cart, best),
        "F: no cart features, 4+11 Nov + sale flag": fit_predict(both, test.assign(sale_week=0),
                                                                 no_cart + ["sale_week"], best),
    }

    # Paired bootstrap: the same resampled users for every option (Poisson weights, about 1 per user)
    rng = np.random.default_rng(0)
    weights = np.zeros((N_BOOT, len(y)), dtype=np.float32)
    for b in range(N_BOOT):
        weights[b] = rng.poisson(1.0, len(y))
    boots, rows = {}, []
    for name, p in scores.items():
        order = np.argsort(-p, kind="stable")
        boots[name] = boot_precisions(order, y, weights)
        rows.append({"option": name, "precision_at_50k": y[order[:BUDGET]].mean(),
                     "expected_over_actual_buyers": p.sum() / y.sum()})
    table = pd.DataFrame(rows)
    ref = boots["B: retrain 4+11 Nov (deployed)"]
    table["diff_vs_B_pts"] = [(boots[n] - ref).mean() * 100 for n in scores]
    table["diff_ci_low"] = [np.quantile(boots[n] - ref, 0.025) * 100 for n in scores]
    table["diff_ci_high"] = [np.quantile(boots[n] - ref, 0.975) * 100 for n in scores]
    table.to_csv(REPORTS / "fix_options_18nov.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"Test week {TEST}: {len(y):,} users, {int(y.sum()):,} buyers")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
