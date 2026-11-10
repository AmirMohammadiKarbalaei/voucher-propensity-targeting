"""
Recalibrate the model's probabilities.

The model ranks users well, but on the test week (18 Nov) its probabilities were about 2x too high.
The experiment design needs calibrated probabilities ("expected buyers = sum of p"), so we fit a small correction on top
of the model: a map from "what the model says" to "what actually happened".

Two situations, compared on the test week:
  1. Before the test week's labels exist: train on 4 Nov, calibrate on 11 Nov (the latest labelled
     week), apply to 18 Nov. Note: 11 Nov is a sale week (Singles' Day), with twice the usual buying.
  2. Once the test week's labels arrive: calibrate on half of the 18 Nov users, check on the other
     half, and swap (cross-fitting), so every user gets a calibrated score from a fit that never saw them.

Methods: Platt (a logistic curve on the model's log-odds: smooth, keeps the ranking exactly) and
isotonic (a step function that only goes up: more flexible, can create ties).

    uv run python modelling/04_calibration.py
"""
import json

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from common import BUDGET, LABEL, REPORTS, ROOT, TEST, TRAIN, VALID, feature_columns, load_gold, split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 0
OUT = ROOT / "data" / "scores"


def lgbm(best: dict) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(n_estimators=best["n_trees"], subsample_freq=1, random_state=SEED, verbose=-1,
                              **best["params"])


def x(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=np.float32)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def platt(fit_p, fit_y):
    m = LogisticRegression(C=1e6).fit(logit(fit_p), fit_y)
    return lambda p: m.predict_proba(logit(p))[:, 1]


def isotonic(fit_p, fit_y):
    m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(fit_p, fit_y)
    return m.predict


def cross_fit(method, p: np.ndarray, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Fit on one half of the users, apply to the other half, and swap."""
    half = np.random.default_rng(seed).random(len(p)) < 0.5
    out = np.empty_like(p)
    out[half] = method(p[~half], y[~half])(p[half])
    out[~half] = method(p[half], y[half])(p[~half])
    return out


def summary(y: np.ndarray, p: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    top = np.argsort(-(p + rng.random(len(p)) * 1e-12))[:BUDGET]
    deciles = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    by_decile = pd.DataFrame({"p": p, "y": y}).groupby(deciles.to_numpy()).mean()
    return {
        "mean_predicted": p.mean(), "actual_rate": y.mean(),
        "predicted_buyers": p.sum(), "actual_buyers": int(y.sum()),
        "brier": np.mean((p - y) ** 2),
        "calibration_error": np.average(np.abs(by_decile["p"] - by_decile["y"])),  # mean gap over deciles
        "top_decile_predicted": by_decile["p"].iloc[-1], "top_decile_actual": by_decile["y"].iloc[-1],
        "precision_at_50k": y[top].mean(),  # ranking check: should barely move
    }


def deciles(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    d = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).to_numpy()
    return pd.DataFrame({"predicted": p, "actual": y}).groupby(d).mean()


def main() -> None:
    best = json.loads((REPORTS / "best_params.json").read_text())["lightgbm"]
    gold = load_gold()
    cols = feature_columns(gold)
    train, valid, test = split(gold, TRAIN), split(gold, VALID), split(gold, TEST)
    y_va, y_te = valid[LABEL].to_numpy(), test[LABEL].to_numpy()

    # Situation 1: model trained on 4 Nov only; calibrator fitted on 11 Nov; judged on 18 Nov
    early = lgbm(best).fit(x(train, cols), train[LABEL])
    p_va_early, p_te_early = early.predict_proba(x(valid, cols))[:, 1], early.predict_proba(x(test, cols))[:, 1]

    # Situation 2: the final model (4 + 11 Nov, as registered and deployed); calibrated on the test week
    both = pd.concat([train, valid], ignore_index=True)
    final = lgbm(best).fit(x(both, cols), both[LABEL])
    p_te = final.predict_proba(x(test, cols))[:, 1]

    rows = {
        "1. 4-Nov model, raw": p_te_early,
        "1. 4-Nov model, Platt fitted on 11 Nov": platt(p_va_early, y_va)(p_te_early),
        "1. 4-Nov model, isotonic fitted on 11 Nov": isotonic(p_va_early, y_va)(p_te_early),
        "2. final model, raw": p_te,
        "2. final model, Platt cross-fitted on 18 Nov": cross_fit(platt, p_te, y_te),
        "2. final model, isotonic cross-fitted on 18 Nov": cross_fit(isotonic, p_te, y_te),
    }
    compare = pd.DataFrame({k: summary(y_te, p) for k, p in rows.items()}).T
    compare.to_csv(REPORTS / "calibration_compare.csv")

    # Reliability diagram: predicted vs actual by decile (perfect = on the diagonal)
    fig, ax = plt.subplots(figsize=(6, 6))
    for name in ["2. final model, raw", "1. 4-Nov model, isotonic fitted on 11 Nov",
                 "2. final model, Platt cross-fitted on 18 Nov"]:
        d = deciles(y_te, rows[name])
        ax.plot(d["predicted"], d["actual"], marker="o", label=name)
    lim = 0.25
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", label="perfect")
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Predicted chance of buying (decile mean)",
           ylabel="Actual share who bought", title="Calibration on the test week (18 Nov)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "calibration_curve.png", dpi=120)

    # Calibrated test-week scores for the experiment design (expected-value maths); data/ is git-ignored
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"user_id": test["user_id"], "label": y_te, "score_raw": p_te,
                  "score_calibrated": rows["2. final model, Platt cross-fitted on 18 Nov"]}
                 ).to_parquet(OUT / f"test_{TEST}.parquet", index=False)

    pd.set_option("display.width", 200)
    print(compare.round(4).to_string())


if __name__ == "__main__":
    main()
