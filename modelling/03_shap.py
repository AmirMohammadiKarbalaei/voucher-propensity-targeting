"""
Explain the winning model with SHAP.

- Global: which features move predictions most, and in which direction (beeswarm, mean |SHAP|).
- Local: why one particular user got a high score (waterfall).

The model is loaded from the MLflow model registry, the same way a deployment would load it.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap

from common import LABEL, MLFLOW_URI, REPORTS, TEST, feature_columns, load_gold, split

SAMPLE = 5_000  # SHAP values for a sample of test users; enough for stable global importance


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    versions = mlflow.MlflowClient().search_model_versions("name='propensity_model'")
    latest = max(int(v.version) for v in versions)
    uri = f"models:/propensity_model/{latest}"
    flavour = mlflow.models.get_model_info(uri).flavors
    loader = next(getattr(mlflow, f) for f in ["lightgbm", "xgboost", "catboost"] if f in flavour)
    model = loader.load_model(uri)
    print(f"Loaded {uri} ({', '.join(f for f in flavour if f != 'python_function')})")

    gold = load_gold()
    cols = feature_columns(gold)
    test = split(gold, TEST)
    x = test[cols].sample(SAMPLE, random_state=0).astype("float32")

    explainer = shap.TreeExplainer(model)
    sv = explainer(x)
    if sv.values.ndim == 3:            # some libraries return one set of values per class
        sv = sv[:, :, 1]

    importance = pd.Series(np.abs(sv.values).mean(axis=0), index=cols).sort_values(ascending=False)
    importance.rename("mean_abs_shap").to_csv(REPORTS / "shap_importance.csv")
    print("\nMean |SHAP| (log-odds), top 10:\n", importance.head(10).round(4).to_string())

    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(REPORTS / "shap_beeswarm.png", dpi=150); plt.close()

    # Local explanation: the highest-scoring user in the sample
    top_user = int(np.argmax(sv.values.sum(axis=1)))
    shap.plots.waterfall(sv[top_user], max_display=12, show=False)
    plt.tight_layout(); plt.savefig(REPORTS / "shap_waterfall_top_user.png", dpi=150); plt.close()
    print("\nSaved reports/shap_beeswarm.png and reports/shap_waterfall_top_user.png")


if __name__ == "__main__":
    main()
