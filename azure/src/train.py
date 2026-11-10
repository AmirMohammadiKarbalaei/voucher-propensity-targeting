"""
Azure ML training job: refit the winning LightGBM (settings tuned locally, in params.json) on the
two training weeks, check it on the test week, and save it as an MLflow model.

    python train.py --data <folder with train.parquet and test.parquet>

Anything written to ./outputs is uploaded to the job automatically, so the model ends up in
outputs/model, ready to register and deploy without a scoring script.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, roc_auc_score

BUDGET = 50_000
LABEL = "label"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    best = json.loads((Path(__file__).parent / "params.json").read_text())
    train = pd.read_parquet(Path(args.data) / "train.parquet")
    test = pd.read_parquet(Path(args.data) / "test.parquet")
    cols = [c for c in train.columns if c != LABEL]
    x_train, x_test = train[cols].astype("float32"), test[cols].astype("float32")  # NaN stays NaN
    print(f"train {len(train):,} rows, test {len(test):,} rows, {len(cols)} features")

    model = lgb.LGBMClassifier(n_estimators=best["n_trees"], subsample_freq=1, random_state=0, verbose=-1,
                               **best["params"])
    model.fit(x_train, train[LABEL])

    y, score = test[LABEL].to_numpy(), model.predict_proba(x_test)[:, 1]
    top = np.argsort(-score)[:BUDGET]
    metrics = {"test_precision_at_50k": y[top].mean(), "test_pr_auc": average_precision_score(y, score),
               "test_roc_auc": roc_auc_score(y, score), "test_base_rate": y.mean()}
    mlflow.log_params({"n_trees": best["n_trees"], **best["params"]})
    mlflow.log_metrics(metrics)
    print(json.dumps(metrics, indent=2))

    # Signature = the column names and types the endpoint will accept.
    # Saved with the scikit-learn flavour so the endpoint returns probabilities (predict_proba), not 0/1.
    proba = model.predict_proba(x_test.head(1000))
    # float64 because JSON numbers arrive as float64; the model itself works the same either way.
    example = x_test.head(1000).astype("float64")
    signature = infer_signature(example, proba)
    # Azure ML's no-code MLflow deployment needs its serving packages listed in the model's own conda
    # env; without azureml-ai-monitoring the scoring container crashes on start-up.
    mlflow.sklearn.save_model(model, "outputs/model", signature=signature, input_example=example.head(3),
                              pyfunc_predict_fn="predict_proba", serialization_format="cloudpickle",
                              extra_pip_requirements=[f"lightgbm=={lgb.__version__}",
                                                      "azureml-ai-monitoring==1.0.0",
                                                      "azureml-inference-server-http==1.5.1"])
    print("saved outputs/model")


if __name__ == "__main__":
    main()
