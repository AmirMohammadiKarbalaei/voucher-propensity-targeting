"""Shared pieces for every model: loading gold, the time-based split, features and metrics."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
REPORTS = ROOT / "reports"
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"

BUDGET = 50_000  # vouchers per week (problem statement)

# Time-based split: always predict a later week from earlier ones
TRAIN, VALID, TEST = "2019-11-04", "2019-11-11", "2019-11-18"

ID_COLS = ["user_id", "cutoff_date"]
LABEL = "label"


def load_gold() -> pd.DataFrame:
    df = pd.read_parquet(GOLD)
    df["cutoff_date"] = df["cutoff_date"].astype(str)
    return df


def split(df: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    return df[df["cutoff_date"] == cutoff].reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS + [LABEL]]


def top_k_metrics(y: np.ndarray, score: np.ndarray, k: int = BUDGET, seed: int = 0) -> dict:
    """Metrics for the decision we actually make: send vouchers to the top k scores."""
    rng = np.random.default_rng(seed)
    tie_break = rng.random(len(score)) * 1e-9          # random order among equal scores
    top = np.argsort(-(score + tie_break))[:k]
    base_rate = y.mean()
    precision = y[top].mean()
    return {
        "precision_at_50k": precision,                  # share of voucher users who buy
        "lift_at_50k": precision / base_rate,           # how many times better than random
        "recall_at_50k": y[top].sum() / y.sum(),        # share of all buyers reached
        "pr_auc": average_precision_score(y, score),    # ranking quality over everyone
        "roc_auc": roc_auc_score(y, score),
        "base_rate": base_rate,
    }
