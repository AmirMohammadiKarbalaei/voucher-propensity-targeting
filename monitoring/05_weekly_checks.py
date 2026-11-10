"""
The weekly monitoring rule, run the way the pipeline would run it.

The rule itself (signals, thresholds, actions) lives in monitoring/checks.py; what a person does
when it fires is in docs/monitoring/runbook.md.

Replay (what the rule would have said in November 2019). The model is trained on the 4 Nov week,
as in 02_prediction_and_performance.py, and that week is the reference:

    uv run python monitoring/05_weekly_checks.py replay

  - scores every week and saves them to data/scores/<week>.parquet (as a scoring job would)
  - day-0 checks for 11 Nov and 18 Nov, day-7 checks once their answers are in
  - fault injection: the 18 Nov input with one feature emptied, to confirm it triggers a STOP
  - writes reports/weekly_checks.csv

Single checks, as the Airflow tasks call them (exit code 1 on STOP, so the send never starts):

    uv run python monitoring/05_weekly_checks.py day0 --week 2019-11-18
    uv run python monitoring/05_weekly_checks.py day7 --week 2019-11-18
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modelling"))
sys.path.insert(0, str(ROOT / "monitoring"))
from checks import STOP, day0_checks, day7_checks, verdict  # noqa: E402
from common import BUDGET, LABEL, REPORTS, feature_columns, load_gold, split  # noqa: E402

SCORES = ROOT / "data" / "scores"
REFERENCE = "2019-11-04"                      # the training week: what "normal" looks like
WEEKS = ["2019-11-04", "2019-11-11", "2019-11-18"]


def score_all_weeks(gold: pd.DataFrame, cols: list[str]) -> None:
    best = json.loads((REPORTS / "best_params.json").read_text())["lightgbm"]
    ref = split(gold, REFERENCE)
    model = lgb.LGBMClassifier(n_estimators=best["n_trees"], subsample_freq=1, random_state=0, verbose=-1,
                               **best["params"]).fit(ref[cols].to_numpy(np.float32), ref[LABEL])
    SCORES.mkdir(parents=True, exist_ok=True)
    for w in WEEKS:
        df = split(gold, w)
        pd.DataFrame({"user_id": df["user_id"],
                      "score": model.predict_proba(df[cols].to_numpy(np.float32))[:, 1]}
                     ).to_parquet(SCORES / f"{w}.parquet", index=False)


def load_scores(week: str, users: pd.Series) -> np.ndarray:
    s = pd.read_parquet(SCORES / f"{week}.parquet")
    return users.to_frame().merge(s, on="user_id", how="left")["score"].to_numpy()


def run_day0(gold, cols, week, cur=None) -> list[dict]:
    ref = split(gold, REFERENCE)
    cur = split(gold, week) if cur is None else cur
    previous = WEEKS[WEEKS.index(week) - 1] if week in WEEKS and week != WEEKS[0] else REFERENCE
    return day0_checks(ref, cur, cols, load_scores(REFERENCE, ref["user_id"]),
                       load_scores(week, cur["user_id"]), normal_rate=ref[LABEL].mean(),
                       last_week_users=len(split(gold, previous)))


def run_day7(gold, week) -> list[dict]:
    def lift(w):
        df = split(gold, w)
        s, y = load_scores(w, df["user_id"]), df[LABEL].to_numpy()
        return y[np.argsort(-s, kind="stable")[:BUDGET]].mean() / y.mean()
    # earlier out-of-sample weeks only: the reference week is in-sample and would flatter the comparison
    earlier = [lift(w) for w in WEEKS if REFERENCE < w < week]
    df = split(gold, week)
    return day7_checks(load_scores(week, df["user_id"]), df[LABEL].to_numpy(), earlier, BUDGET)


def show(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title}: {verdict(rows)} ===")
    for r in rows:
        if r["level"] != "OK" or r["stage"] != "health":      # healthy health checks: keep the print short
            print(f"  [{r['level']:5}] {r['check']}: {r['value']}  (ok if {r['threshold']})"
                  + (f"\n          -> {r['action']}" if r["level"] != "OK" else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["replay", "day0", "day7"])
    ap.add_argument("--week", default="2019-11-18")
    args = ap.parse_args()

    gold = load_gold()
    cols = feature_columns(gold)

    if args.mode == "day0":
        rows = run_day0(gold, cols, args.week)
        show(f"{args.week} day 0", rows)
        sys.exit(1 if verdict(rows) == STOP else 0)
    if args.mode == "day7":
        show(f"{args.week} day 7", run_day7(gold, args.week))
        return

    score_all_weeks(gold, cols)
    log = []
    for w in WEEKS[1:]:
        for stage, rows in [("day 0", run_day0(gold, cols, w)), ("day 7", run_day7(gold, w))]:
            show(f"{w} {stage}", rows)
            log += [{"week": w, "run": stage, **r} for r in rows]

    broken = split(gold, "2019-11-18").assign(n_events_7d=np.nan)      # fault injection
    rows = run_day0(gold, cols, "2019-11-18", cur=broken)
    show("2019-11-18 day 0, n_events_7d emptied (fault injection)", rows)
    log += [{"week": "2019-11-18 (fault injection)", "run": "day 0", **r} for r in rows]

    pd.DataFrame(log).to_csv(REPORTS / "weekly_checks.csv", index=False)


if __name__ == "__main__":
    main()
