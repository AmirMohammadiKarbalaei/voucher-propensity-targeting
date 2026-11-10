"""
Watch the model's outputs (straight away) and its accuracy (7 days later).

Replay of a realistic timeline: a model trained on the 4 Nov week is used to score 11 Nov and 18 Nov.
  - Day 0 (no labels yet): score distribution vs the training week -> prediction drift (PSI of scores),
    mean score, how many users clear the old top-50k cut-off.
  - Day 7 (labels arrive): actual buy rate, expected vs actual buyers (calibration), precision@50k,
    PR-AUC, lift.

4 Nov is the training week, so its numbers are in-sample (a bit flattering); it is the reference.

    uv run python monitoring/02_prediction_and_performance.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modelling"))
sys.path.insert(0, str(ROOT / "monitoring"))
from common import BUDGET, LABEL, REPORTS, feature_columns, load_gold, split, top_k_metrics  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WEEKS = ["2019-11-04", "2019-11-11", "2019-11-18"]
TRAIN_WEEK = WEEKS[0]


def score_psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI of the model's scores, using the reference week's score deciles as buckets."""
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-6, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def main() -> None:
    best = json.loads((REPORTS / "best_params.json").read_text())["lightgbm"]
    gold = load_gold()
    cols = feature_columns(gold)
    train = split(gold, TRAIN_WEEK)
    model = lgb.LGBMClassifier(n_estimators=best["n_trees"], subsample_freq=1, random_state=0, verbose=-1,
                               **best["params"]).fit(train[cols].to_numpy(np.float32), train[LABEL])

    scores = {w: model.predict_proba(split(gold, w)[cols].to_numpy(np.float32))[:, 1] for w in WEEKS}
    ref = scores[TRAIN_WEEK]
    old_cutoff = np.sort(ref)[-BUDGET]                      # score of the 50,000th user in the training week

    rows = []
    for w in WEEKS:
        p, y = scores[w], split(gold, w)[LABEL].to_numpy()
        m = top_k_metrics(y, p)
        rows.append({
            "week": w, "users": len(p),
            # Day 0: no labels needed
            "mean_score": p.mean(), "score_psi_vs_train": score_psi(ref, p),
            "users_above_old_cutoff": int((p >= old_cutoff).sum()),
            # Day 7: needs labels
            "actual_rate": y.mean(), "expected_buyers": p.sum(), "actual_buyers": int(y.sum()),
            "expected_over_actual": p.sum() / y.sum(),
            "precision_at_50k": m["precision_at_50k"], "lift_at_50k": m["lift_at_50k"], "pr_auc": m["pr_auc"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "performance_by_week.csv", index=False)
    pd.set_option("display.width", 200)
    print("Model trained on", TRAIN_WEEK, "(that week is in-sample)\n")
    print(table.T.to_string(header=False, float_format=lambda v: f"{v:.4g}"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(WEEKS))
    axes[0].bar(x - 0.2, table.mean_score * 100, 0.4, label="model's average score (day 0)")
    axes[0].bar(x + 0.2, table.actual_rate * 100, 0.4, label="actual buy rate (day 7)")
    axes[0].set(xticks=x, xticklabels=["4 Nov\n(training)", "11 Nov\n(sale week)", "18 Nov"],
                ylabel="%", title="Predicted vs actual buying")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, table.precision_at_50k * 100, color="tab:green")
    axes[1].set(xticks=x, xticklabels=["4 Nov\n(training)", "11 Nov\n(sale week)", "18 Nov"],
                ylabel="% of top 50,000 who bought", title="Ranking quality (precision@50k)")
    fig.suptitle("Model trained on 4 Nov, monitored over the next two weeks")
    fig.tight_layout()
    fig.savefig(REPORTS / "performance_by_week.png", dpi=120)


if __name__ == "__main__":
    main()
