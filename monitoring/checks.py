"""
The weekly monitoring rule: every signal, threshold and action in one place.

Three outcomes per check:
  STOP   the data is broken; the ranking can't be trusted. Don't send; a person checks today.
  ALERT  something changed; send as usual (the ranking held up under drift in the November replay), a person
         looks this week and the money estimates are anchored / reported as a range.
  OK     nothing to do.

Day 0 (Monday, before the voucher send): data health, input drift, output drift.
Day 7 (the following Monday, answers in): calibration and ranking quality.
Thresholds come from what we measured on the November weeks; see docs/monitoring/runbook.md.
"""
import numpy as np
import pandas as pd

OK, ALERT, STOP = "OK", "ALERT", "STOP"

# ---- The rule. Change thresholds here and nowhere else. ----
ROWS_CHANGE_STOP = 0.30          # users to score vs LAST week: more than +/-30% = broken load
                                 # (vs a fixed old week, normal growth tripped it: +33% by 23 Dec)
MISSING_JUMP_STOP = 0.20         # a feature's missing share rises by more than 20 points = broken
FEATURE_PSI_ALERT = 0.25         # "big shift" in an input (carts on 18 Nov: 0.27-0.32)
SCORE_PSI_ALERT = 0.10           # the scores themselves have moved
MEAN_SCORE_RATIO = (0.7, 1.3)    # mean score / normal buy rate; outside = levels suspect, anchor them
EXP_ACT_RATIO = (0.8, 1.25)      # expected / actual buyers (day 7); outside = refit the translator
LIFT_DROP_RETRAIN = 0.25         # lift@budget more than 25% below the median of earlier weeks = retrain


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population stability index: how much users moved between the reference's buckets.

    Buckets: missing, exactly 0, then 10 quantile buckets of the reference's non-zero values.
    Zero gets its own bucket because most users have 0 carts/purchases; plain deciles would put
    everyone in one bucket and hide a 12x jump in carts (a bug found in the first version).
    """
    ref, cur = ref.astype(float), cur.astype(float)
    nonzero = ref[(ref != 0) & ref.notna()]
    edges = np.unique(np.quantile(nonzero, np.linspace(0, 1, bins + 1))) if len(nonzero) else np.array([])
    edges = np.array([-np.inf, *edges[1:-1], np.inf])   # open ends; a 0/1 flag ends up with one "1" bucket

    def shares(x: pd.Series) -> np.ndarray:
        nz = x[(x != 0) & x.notna()]
        counts = np.concatenate([[x.isna().sum(), (x == 0).sum()], np.histogram(nz, bins=edges)[0]])
        return np.clip(counts / len(x), 1e-6, None)

    r, c = shares(ref), shares(cur)
    return float(np.sum((c - r) * np.log(c / r)))


def _row(stage, check, value, threshold, level, action):
    return {"stage": stage, "check": check, "value": value, "threshold": threshold, "level": level, "action": action}


def day0_checks(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str], ref_scores: np.ndarray,
                cur_scores: np.ndarray, normal_rate: float, last_week_users: int | None = None) -> list[dict]:
    """Checks that need no answers: run on Monday morning before the send.

    last_week_users: users scored last week; the row-count check compares with it (falls back to the
    reference week when there is no previous week).
    """
    out = []
    # 1. Data health: is the input usable at all?
    base = last_week_users or len(ref)
    change = len(cur) / base - 1
    out.append(_row("health", "users to score vs last week", f"{change:+.1%}", f"within +/-{ROWS_CHANGE_STOP:.0%}",
                    STOP if abs(change) > ROWS_CHANGE_STOP else OK, "check the gold load (missing or doubled data)"))
    dups = int(cur["user_id"].duplicated().sum())
    out.append(_row("health", "duplicate users", dups, "0", STOP if dups else OK, "check the gold load"))
    for f in features:
        jump = cur[f].isna().mean() - ref[f].isna().mean()
        if jump > MISSING_JUMP_STOP:
            out.append(_row("health", f"missing share up: {f}", f"{jump:+.1%}", f"< +{MISSING_JUMP_STOP:.0%}", STOP,
                            "find the broken upstream column before sending"))
        if ref[f].nunique() > 1 and cur[f].nunique() <= 1:
            out.append(_row("health", f"feature became constant: {f}", cur[f].iloc[0], "varies", STOP,
                            "find the broken upstream column before sending"))
    bad = int((~np.isfinite(cur_scores) | (cur_scores < 0) | (cur_scores > 1)).sum())
    out.append(_row("health", "invalid scores", bad, "0", STOP if bad else OK, "check the scoring step"))

    # 2. Input drift: one row per feature over the threshold
    drifted = [(f, psi(ref[f], cur[f])) for f in features]
    drifted = sorted([d for d in drifted if d[1] > FEATURE_PSI_ALERT], key=lambda d: -d[1])
    out.append(_row("inputs", "features with PSI > threshold", ", ".join(f"{f} {p:.2f}" for f, p in drifted) or "none",
                    f"PSI <= {FEATURE_PSI_ALERT}", ALERT if drifted else OK,
                    "send; find out why these inputs moved (logging change? real behaviour?)"))

    # 3. Output drift
    s_psi = psi(pd.Series(ref_scores), pd.Series(cur_scores))
    out.append(_row("outputs", "score PSI", round(s_psi, 3), f"<= {SCORE_PSI_ALERT}",
                    ALERT if s_psi > SCORE_PSI_ALERT else OK, "send; levels may be off"))
    ratio = cur_scores.mean() / normal_rate
    lo, hi = MEAN_SCORE_RATIO
    out.append(_row("outputs", "mean score / normal buy rate", round(ratio, 2), f"{lo}-{hi}",
                    ALERT if not lo <= ratio <= hi else OK,
                    "send; anchor the money estimates to the normal rate and report a range"))
    return out


def day7_checks(scores: np.ndarray, y: np.ndarray, earlier_lifts: list[float], budget: int) -> list[dict]:
    """Checks that need the answers: run 7 days after the send."""
    out = []
    ratio = scores.sum() / y.sum()
    lo, hi = EXP_ACT_RATIO
    out.append(_row("results", "expected / actual buyers", round(ratio, 2), f"{lo}-{hi}",
                    ALERT if not lo <= ratio <= hi else OK,
                    "refit the translator on this week if it was a normal week; else note it as unusual"))
    top = np.argsort(-scores, kind="stable")[:budget]
    lift = y[top].mean() / y.mean()
    if earlier_lifts:
        ref = float(np.median(earlier_lifts))
        drop = 1 - lift / ref
        out.append(_row("results", f"lift@{budget:,} vs earlier weeks ({ref:.1f})", round(lift, 1),
                        f"drop <= {LIFT_DROP_RETRAIN:.0%}", ALERT if drop > LIFT_DROP_RETRAIN else OK,
                        "RETRAIN: the ranking itself got worse"))
    else:
        out.append(_row("results", f"lift@{budget:,}", round(lift, 1), "no earlier weeks yet", OK,
                        "keep as the first comparison point"))
    return out


def verdict(rows: list[dict]) -> str:
    levels = {r["level"] for r in rows}
    return STOP if STOP in levels else ALERT if ALERT in levels else OK
