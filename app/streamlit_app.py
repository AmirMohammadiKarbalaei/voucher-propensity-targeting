"""
Demo app: send one user's features to the Azure ML online endpoint and show the chance they buy in
the next 7 days.

    uv run streamlit run app/streamlit_app.py

The endpoint address and key are read from environment variables (or a git-ignored .env file in the
repo root), never from code:
    AZURE_ML_ENDPOINT_URL=https://<endpoint>.uksouth.inference.ml.azure.com/score
    AZURE_ML_ENDPOINT_KEY=<primary key from the endpoint's Consume tab>
"""
import json
import os
import time
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = json.loads((ROOT / "azure" / "sample_request.json").read_text())
COLUMNS = SAMPLE["input_data"]["columns"]
PRESETS = dict(zip(["Quiet user (one visit, 9 days ago)", "Active this week, carted and bought",
                    "Heavy user (4,500 events in 30 days)"], SAMPLE["input_data"]["data"]))
BASE_RATE = 0.0204  # share of users who bought in the test week (18 Nov)

# The features people understand at a glance; the rest keep the preset's values.
EDITABLE = {
    "days_since_last_event": ("Days since last visit", 0.0, 30.0),
    "days_since_last_purchase": ("Days since last purchase", 0.0, 30.0),
    "n_events_7d": ("Events in last 7 days", 0.0, 2000.0),
    "n_carts_7d": ("Add-to-carts in last 7 days", 0.0, 100.0),
    "n_purchases_30d": ("Purchases in last 30 days", 0.0, 50.0),
    "carted_not_bought_7d": ("Carted but didn't buy in last 7 days (0/1)", 0.0, 1.0),
}


def load_env() -> None:
    """Read KEY=value lines from ROOT/.env into the environment (only if not already set)."""
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


def score(url: str, key: str, row: list[float]) -> tuple[float, float]:
    body = {"input_data": {"columns": COLUMNS, "index": [0], "data": [row]}}
    start = time.perf_counter()
    resp = requests.post(url, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    resp.raise_for_status()
    return resp.json()[0][1], (time.perf_counter() - start) * 1000


load_env()
url, key = os.environ.get("AZURE_ML_ENDPOINT_URL"), os.environ.get("AZURE_ML_ENDPOINT_KEY")

st.set_page_config(page_title="Purchase propensity", page_icon="🛒")
st.title("Will this user buy in the next 7 days?")
st.caption("LightGBM model trained on REES46 e-commerce events (Nov 2019), served from an Azure ML "
           "managed online endpoint. Personal portfolio project.")

if not url or not key:
    st.error("Set AZURE_ML_ENDPOINT_URL and AZURE_ML_ENDPOINT_KEY (environment or .env) first.")
    st.stop()

preset = st.selectbox("Start from a real test-week user", list(PRESETS))
row = dict(zip(COLUMNS, PRESETS[preset]))

with st.form("features"):
    cols = st.columns(2)
    for i, (name, (label, lo, hi)) in enumerate(EDITABLE.items()):
        value = float(min(max(row[name], lo), hi))
        if name == "carted_not_bought_7d":
            row[name] = float(cols[i % 2].checkbox(label, value=bool(value)))
        else:
            row[name] = cols[i % 2].number_input(label, lo, hi, value)
    submitted = st.form_submit_button("Score this user")

if submitted:
    try:
        p, ms = score(url, key, [row[c] for c in COLUMNS])
    except requests.RequestException as err:
        st.error(f"Endpoint call failed: {err}")
        st.stop()
    a, b, c = st.columns(3)
    a.metric("Model score", f"{p:.1%}")
    b.metric("vs average user", f"{p / BASE_RATE:.1f}×")
    c.metric("Response time", f"{ms:.0f} ms")
    st.info("The score is for **ranking** (who goes in the top 50,000 for vouchers). On the test week "
            "the raw probabilities were about 2× too high, so read 20% as 'roughly 10%' until the "
            "model is recalibrated.")
    with st.expander("Request sent to the endpoint"):
        st.json({"input_data": {"columns": COLUMNS, "data": [[row[c] for c in COLUMNS]]}})
