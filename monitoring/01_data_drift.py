"""
Data drift. Have the model's inputs changed since the training week?

Reference = 4 Nov (the first training week). Current = 11 Nov (sale week) and 18 Nov (test week).
Two ways of measuring, side by side:
  - Evidently DataDriftPreset (0.7): a statistical test / distance per feature, HTML report for people.
  - PSI (population stability index), computed by hand: one number per feature, easy to explain
    and the usual industry rule of thumb (< 0.1 stable, 0.1-0.25 watch, > 0.25 big shift).

Evidently runs on a random 100,000 users per week (it slows down on millions); PSI uses everyone.

    uv run python monitoring/01_data_drift.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modelling"))
sys.path.insert(0, str(ROOT / "monitoring"))
from checks import psi  # noqa: E402  (shared with the weekly checks)
from common import REPORTS, feature_columns, load_gold, split  # noqa: E402

DOCS = ROOT / "docs" / "monitoring"
REFERENCE, CURRENT = "2019-11-04", ["2019-11-11", "2019-11-18"]
SAMPLE = 100_000


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    gold = load_gold()
    cols = feature_columns(gold)
    ref = split(gold, REFERENCE)[cols]

    rows = []
    for week in CURRENT:
        cur = split(gold, week)[cols]
        snap = Report([DataDriftPreset()]).run(current_data=cur.sample(SAMPLE, random_state=0),
                                               reference_data=ref.sample(SAMPLE, random_state=0))
        snap.save_html(str(DOCS / f"data_drift_{week}.html"))
        evid = {m["config"]["column"]: (m["config"]["method"], m["value"])
                for m in snap.dict()["metrics"] if m["config"].get("column")}
        for c in cols:
            method, value = evid[c]
            rows.append({"week": week, "feature": c, "mean_reference": ref[c].mean(), "mean_current": cur[c].mean(),
                         "psi": psi(ref[c], cur[c]), "evidently_method": method, "evidently_score": value})

    table = pd.DataFrame(rows)
    table["evidently_drift"] = table.evidently_score > 0.1            # Evidently's default threshold here
    table["psi_level"] = pd.cut(table.psi, [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "watch", "big shift"])
    table.to_csv(REPORTS / "drift_by_feature.csv", index=False)

    pd.set_option("display.width", 200)
    for week in CURRENT:
        t = table[table.week == week].sort_values("psi", ascending=False)
        print(f"\n=== {REFERENCE} (reference) vs {week} ===  users: {len(ref):,} vs {len(split(gold, week)):,}")
        print(f"Evidently flags {t.evidently_drift.sum()} of {len(t)} features; "
              f"PSI: {(t.psi_level == 'big shift').sum()} big shift, {(t.psi_level == 'watch').sum()} watch")
        print(t[["feature", "mean_reference", "mean_current", "psi", "psi_level", "evidently_score"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
