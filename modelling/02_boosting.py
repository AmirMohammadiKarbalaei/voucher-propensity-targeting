"""
XGBoost vs LightGBM vs CatBoost, compared fairly.

Fair = same features, same split, same tuning budget (Optuna trials), same selection metric.

    uv run python 02_boosting.py tune    # Optuna on validation (11 Nov); saves reports/best_params.json
    uv run python 02_boosting.py final   # refit on 4 + 11 Nov, score the test week (18 Nov) once

Boosted trees handle missing values and skewed counts themselves, so they get the raw gold features
(no filling, no logs), unlike logistic regression.
"""
import json
import sys
import time

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from common import LABEL, REPORTS, TEST, TRAIN, VALID, feature_columns, load_gold, split, top_k_metrics

N_TRIALS = 15            # same budget for every library
TUNE_SAMPLE = 1_000_000  # tune on 1M of the 3M training users to keep each trial quick
EARLY_STOP = 50          # stop adding trees when validation loss hasn't improved for 50 rounds
MAX_TREES = 2000
SEED = 0


def suggest(trial: optuna.Trial, name: str) -> dict:
    """The search space for each library: the main knobs that control tree size, learning speed and regularisation."""
    lr = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
    if name == "lightgbm":
        return {"learning_rate": lr,
                "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if name == "xgboost":
        return {"learning_rate": lr,
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    return {"learning_rate": lr,                                        # catboost
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
            "rsm": trial.suggest_float("rsm", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0)}


def make_model(name: str, params: dict, n_trees: int | None = None):
    """n_trees=None: up to MAX_TREES with early stopping (tuning). A number: exactly that many trees (final refit)."""
    early_stop = n_trees is None
    n_trees = n_trees or MAX_TREES
    # Early stopping watches PR-AUC (ranking), not log loss: the buying rate differs between weeks
    # (2.3% vs 4.3%), which makes log loss worsen even while the ranking is still improving.
    if name == "lightgbm":
        return lgb.LGBMClassifier(n_estimators=n_trees, subsample_freq=1, metric="average_precision",
                                  random_state=SEED, verbose=-1, **params)
    if name == "xgboost":
        return xgb.XGBClassifier(n_estimators=n_trees, tree_method="hist", eval_metric="aucpr",
                                 early_stopping_rounds=EARLY_STOP if early_stop else None,
                                 random_state=SEED, n_jobs=-1, **params)
    return cb.CatBoostClassifier(iterations=n_trees, bootstrap_type="Bernoulli", eval_metric="PRAUC",
                                 random_seed=SEED, verbose=False, thread_count=-1, **params)


def fit(name: str, model, x_tr, y_tr, x_va=None, y_va=None):
    """Fit with early stopping when a validation set is given; return the model and the number of trees used."""
    if x_va is None:
        model.fit(x_tr, y_tr)
        return model, None
    if name == "lightgbm":
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        return model, model.best_iteration_
    if name == "xgboost":
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
        return model, model.best_iteration + 1
    model.fit(x_tr, y_tr, eval_set=(x_va, y_va), early_stopping_rounds=EARLY_STOP)
    return model, model.get_best_iteration() + 1


def as_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=np.float32)  # NaN stays NaN: the trees learn where missing values should go


def tune() -> None:
    gold = load_gold()
    cols = feature_columns(gold)
    train = split(gold, TRAIN).sample(TUNE_SAMPLE, random_state=SEED)
    valid = split(gold, VALID)
    x_tr, y_tr = as_matrix(train, cols), train[LABEL].to_numpy()
    x_va, y_va = as_matrix(valid, cols), valid[LABEL].to_numpy()

    # Resumable: models already tuned (saved after each one) are skipped
    path = REPORTS / "best_params.json"
    best = json.loads(path.read_text()) if path.exists() else {}
    for name in ["lightgbm", "xgboost", "catboost"]:
        if name in best:
            print(name, "already tuned, skipping", flush=True)
            continue

        def objective(trial):
            model, n_trees = fit(name, make_model(name, suggest(trial, name)), x_tr, y_tr, x_va, y_va)
            trial.set_user_attr("n_trees", int(n_trees))
            return average_precision_score(y_va, model.predict_proba(x_va)[:, 1])  # selection metric: PR-AUC

        start = time.time()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=N_TRIALS)
        best[name] = {"params": study.best_params, "n_trees": study.best_trial.user_attrs["n_trees"],
                      "valid_pr_auc": study.best_value, "tuning_minutes": round((time.time() - start) / 60, 1)}
        print(name, json.dumps(best[name]), flush=True)
        REPORTS.mkdir(exist_ok=True)
        path.write_text(json.dumps(best, indent=2))


def bootstrap_precision(y: np.ndarray, scores: dict, k: int = 50_000, reps: int = 200) -> pd.DataFrame:
    """Resample users with replacement and recompute precision in the top k for every model on the SAME resample,
    so differences between models get a confidence interval too (a paired bootstrap)."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    draws = {m: np.empty(reps) for m in scores}
    for r in range(reps):
        idx = rng.integers(0, n, n)
        y_r, noise = y[idx], rng.random(n) * 1e-9               # noise = random tie-break
        for m, s in scores.items():
            top = np.argpartition(-(s[idx] + noise), k - 1)[:k]
            draws[m][r] = y_r[top].mean()
    return pd.DataFrame(draws)


def final() -> None:
    import importlib.util

    import mlflow
    import statsmodels.api as sm

    from common import MLFLOW_URI

    # Reuse the logistic regression preparation from 01_baselines_logreg.py
    spec = importlib.util.spec_from_file_location("step1", REPORTS.parent / "modelling" / "01_baselines_logreg.py")
    step1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step1)

    best = json.loads((REPORTS / "best_params.json").read_text())
    gold = load_gold()
    cols = feature_columns(gold)
    train, valid, test = split(gold, TRAIN), split(gold, VALID), split(gold, TEST)
    train_valid = pd.concat([train, valid], ignore_index=True)
    y_va, y_te = valid[LABEL].to_numpy(), test[LABEL].to_numpy()

    def logistic_scores(fit_df: pd.DataFrame, score_df: pd.DataFrame) -> np.ndarray:
        x_fit, x_score = step1.prepare(fit_df), step1.prepare(score_df)
        c = step1.CONTINUOUS
        mean, std = x_fit[c].mean(), x_fit[c].std()
        x_fit[c], x_score[c] = (x_fit[c] - mean) / std, (x_score[c] - mean) / std
        sample = x_fit.sample(500_000, random_state=SEED)
        m = sm.Logit(fit_df.loc[sample.index, LABEL], sm.add_constant(sample)).fit(disp=False, maxiter=200)
        return m.predict(sm.add_constant(x_score)).to_numpy()

    def rule_scores(df: pd.DataFrame) -> np.ndarray:
        return (df["carted_not_bought_7d"] * (100 - df["days_since_last_cart"].fillna(30))).to_numpy()

    # 1. Validation check: each tuned model refit on all of 4 Nov, scored on 11 Nov (comparable to the baselines)
    valid_scores = {"B1 current rule": rule_scores(valid), "Logistic regression": logistic_scores(train, valid)}
    for name, b in best.items():
        model = make_model(name, b["params"], n_trees=b["n_trees"])
        model.fit(as_matrix(train, cols), train[LABEL].to_numpy())
        valid_scores[name] = model.predict_proba(as_matrix(valid, cols))[:, 1]
    valid_metrics = pd.DataFrame({m: top_k_metrics(y_va, s) for m, s in valid_scores.items()}).T

    # The winner is chosen on VALIDATION PR-AUC; the test week only reports how it does
    winner = max(best, key=lambda m: valid_metrics.loc[m, "pr_auc"])

    # 2. Final: refit on 4 + 11 Nov, score the test week (18 Nov) once
    test_scores = {"B0 random": np.random.default_rng(1).random(len(test)),
                   "B1 current rule": rule_scores(test),
                   "Logistic regression": logistic_scores(train_valid, test)}
    models = {}
    for name, b in best.items():
        model = make_model(name, b["params"], n_trees=b["n_trees"])
        model.fit(as_matrix(train_valid, cols), train_valid[LABEL].to_numpy())
        models[name] = model
        test_scores[name] = model.predict_proba(as_matrix(test, cols))[:, 1]
    test_metrics = pd.DataFrame({m: top_k_metrics(y_te, s) for m, s in test_scores.items()}).T
    # Brier only means something for real probabilities (not the random or rule scores)
    test_metrics["brier"] = [np.nan if m.startswith("B") else np.mean((s - y_te) ** 2) for m, s in test_scores.items()]

    # 3. Uncertainty: 95% intervals for precision in the top 50k, and for the winner's gain over the others
    boot = bootstrap_precision(y_te, {m: s for m, s in test_scores.items() if m != "B0 random"})
    ci = boot.quantile([0.025, 0.975]).T.rename(columns={0.025: "ci_low", 0.975: "ci_high"})
    diffs = pd.DataFrame({f"{winner} minus {m}": boot[winner] - boot[m] for m in boot if m != winner})
    diff_ci = pd.DataFrame({"mean": diffs.mean(), "ci_low": diffs.quantile(0.025), "ci_high": diffs.quantile(0.975)})

    # 4. Calibration of the winner: predicted vs actual buying rate, in 10 groups of predicted probability
    s = test_scores[winner]
    bins = pd.qcut(s, 10, labels=False, duplicates="drop")
    calibration = pd.DataFrame({"predicted": s, "actual": y_te, "bin": bins}).groupby("bin").mean()

    for name, frame in {"valid_metrics_all": valid_metrics, "test_metrics": test_metrics,
                        "test_precision_ci": ci, "test_winner_gain_ci": diff_ci,
                        "test_calibration_winner": calibration}.items():
        frame.to_csv(REPORTS / f"{name}.csv")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("propensity")
    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name} (final)"):
            mlflow.log_params({**best[name]["params"], "n_trees": best[name]["n_trees"],
                               "train_cutoffs": f"{TRAIN},{VALID}", "test_cutoff": TEST})
            mlflow.log_metrics({f"valid_{k}": v for k, v in valid_metrics.loc[name].items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.loc[name].items() if pd.notna(v)})
            flavour = {"lightgbm": mlflow.lightgbm, "xgboost": mlflow.xgboost, "catboost": mlflow.catboost}[name]
            flavour.log_model(model, name="model",
                              registered_model_name="propensity_model" if name == winner else None)

    pd.set_option("display.width", 220)
    print("Validation (11 Nov), models fit on 4 Nov:\n", valid_metrics.round(4).to_string())
    print(f"\nWinner on validation PR-AUC: {winner}")
    print("\nTest (18 Nov), models fit on 4 + 11 Nov:\n", test_metrics.round(4).to_string())
    print("\n95% CI, precision in top 50k:\n", ci.round(4).to_string())
    print("\n95% CI, winner's gain:\n", diff_ci.round(4).to_string())
    print("\nCalibration (winner):\n", calibration.round(4).to_string())


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "tune"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if stage == "tune":
        tune()
    elif stage == "final":
        final()
