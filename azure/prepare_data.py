"""
Prepare the files to upload to Azure ML (run locally, once):

- data/azure/ecom_gold/train.parquet   the two training weeks (4 + 11 Nov): features + label
- data/azure/ecom_gold/test.parquet    the test week (18 Nov), to check Azure reproduces the local result
- azure/sample_request.json            three test-week users, in the format the endpoint expects

    uv run python azure/prepare_data.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modelling"))
from common import LABEL, TEST, TRAIN, VALID, feature_columns, load_gold, split  # noqa: E402

OUT = ROOT / "data" / "azure" / "ecom_gold"


def main() -> None:
    gold = load_gold()
    cols = feature_columns(gold)
    OUT.mkdir(parents=True, exist_ok=True)

    train = gold[gold["cutoff_date"].isin([TRAIN, VALID])]
    test = split(gold, TEST)
    train[cols + [LABEL]].to_parquet(OUT / "train.parquet", index=False)
    test[cols + [LABEL]].to_parquet(OUT / "test.parquet", index=False)
    print(f"train {len(train):,} rows, test {len(test):,} rows -> {OUT}")

    # Three users with no missing values (a low, a middle and a high-activity user), as an endpoint request
    complete = test.dropna(subset=cols).sort_values("n_events_30d")
    users = complete.iloc[[0, len(complete) // 2, -1]][cols]
    request = {"input_data": {"columns": cols, "index": [0, 1, 2], "data": users.astype(float).values.tolist()}}
    (ROOT / "azure" / "sample_request.json").write_text(json.dumps(request, indent=2))
    print("wrote azure/sample_request.json")


if __name__ == "__main__":
    main()
