"""Does the booking advice actually hold up?

The headline accuracy says how close a *single* fare estimate lands. That is not the
claim the advisor makes. The advisor says "option A is cheaper than option B" and
"fly on the 26th rather than the 15th", which are *comparisons*. A model with a
+-Rs.540 average error can still rank two itineraries correctly almost every time if
its errors are correlated across similar flights, and it can also be useless at it.
This script measures which, using the stored out-of-fold predictions.

Run:  python src/validate_savings.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from config import REPORT_DIR
from data_prep import prepare

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RNG = np.random.default_rng(42)
MAX_PAIRS_PER_GROUP = 60
BUCKETS = [0, 250, 500, 1000, 2000, np.inf]
BUCKET_LABELS = [
    "under Rs.250",
    "Rs.250-500",
    "Rs.500-1,000",
    "Rs.1,000-2,000",
    "over Rs.2,000",
]


def _pairs_within(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sample within-group index pairs for every group defined by `keys`."""
    rows = []
    for _, idx in frame.groupby(keys, sort=False).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        n_pairs = min(MAX_PAIRS_PER_GROUP, len(idx) * (len(idx) - 1) // 2)
        left = RNG.choice(idx, size=n_pairs)
        right = RNG.choice(idx, size=n_pairs)
        keep = left != right
        rows.append(np.column_stack([left[keep], right[keep]]))
    if not rows:
        return pd.DataFrame(columns=["i", "j"])
    stacked = np.vstack(rows)
    return pd.DataFrame({"i": stacked[:, 0], "j": stacked[:, 1]})


def _score(frame: pd.DataFrame, pairs: pd.DataFrame, label: str) -> dict:
    true_i = frame["y_true"].to_numpy()[pairs["i"]]
    true_j = frame["y_true"].to_numpy()[pairs["j"]]
    pred_i = frame["y_pred"].to_numpy()[pairs["i"]]
    pred_j = frame["y_pred"].to_numpy()[pairs["j"]]

    true_gap = true_i - true_j
    pred_gap = pred_i - pred_j
    # Ties in the truth carry no orderable information.
    orderable = true_gap != 0
    concordant = np.sign(pred_gap[orderable]) == np.sign(true_gap[orderable])

    result = {
        "comparison": label,
        "pairs": int(orderable.sum()),
        "ranking_accuracy": float(concordant.mean() * 100),
        "MAE_on_difference": float(np.mean(np.abs(pred_gap - true_gap))),
        "MAE_on_level": float(
            np.mean(np.abs(np.concatenate([pred_i - true_i, pred_j - true_j])))
        ),
    }
    # If errors were independent, the error on a difference would be ~1.41x the error
    # on a level. Lower than that means errors cancel and comparisons are safer.
    result["error_inflation"] = result["MAE_on_difference"] / result["MAE_on_level"]
    return result


def _by_confidence(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    true_gap = frame["y_true"].to_numpy()[pairs["i"]] - frame["y_true"].to_numpy()[pairs["j"]]
    pred_gap = frame["y_pred"].to_numpy()[pairs["i"]] - frame["y_pred"].to_numpy()[pairs["j"]]
    orderable = true_gap != 0
    true_gap, pred_gap = true_gap[orderable], pred_gap[orderable]

    band = pd.cut(np.abs(pred_gap), bins=BUCKETS, labels=BUCKET_LABELS)
    table = pd.DataFrame(
        {
            "predicted_gap": band,
            "correct": np.sign(pred_gap) == np.sign(true_gap),
        }
    )
    out = table.groupby("predicted_gap", observed=True)["correct"].agg(["count", "mean"])
    out["mean"] = (out["mean"] * 100).round(1)
    return out.rename(columns={"count": "pairs", "mean": "ranking_accuracy_pct"})


def main() -> None:
    oof = pd.read_csv(REPORT_DIR / "oof_stacked.csv")
    train, _ = prepare(drop_duplicates=True)
    if len(train) != len(oof):
        raise RuntimeError(
            "oof_stacked.csv does not line up with the training data; rerun train.py"
        )

    frame = train.reset_index(drop=True).join(oof)

    print("=" * 78)
    print("Do the booking recommendations hold up?")
    print("=" * 78)

    scenarios = [
        (
            "same route + date, different itinerary",
            ["Source_City", "Destination_City", "Date_of_Journey"],
        ),
        (
            "same flight, different travel date",
            ["Airline", "Route", "Dep_Time"],
        ),
    ]

    summaries = []
    band_tables = []
    for label, keys in scenarios:
        pairs = _pairs_within(frame, keys)
        if pairs.empty:
            continue
        summary = _score(frame, pairs, label)
        summaries.append(summary)
        bands = _by_confidence(frame, pairs).reset_index()
        bands.insert(0, "comparison", label)
        band_tables.append(bands)
        print(f"\n{label}")
        print(f"  comparable pairs      {summary['pairs']:,}")
        print(f"  ranking accuracy      {summary['ranking_accuracy']:.1f}%")
        print(f"  error on a single fare  Rs.{summary['MAE_on_level']:,.0f}")
        print(f"  error on the gap        Rs.{summary['MAE_on_difference']:,.0f} "
              f"({summary['error_inflation']:.2f}x the single-fare error; "
              f"1.41x would mean errors are independent)")
        print("  accuracy by how big the predicted gap is:")
        print(_by_confidence(frame, pairs).to_string().replace("\n", "\n    "))

    pd.DataFrame(summaries).to_csv(REPORT_DIR / "savings_validation.csv", index=False)
    pd.concat(band_tables, ignore_index=True).to_csv(
        REPORT_DIR / "savings_confidence_bands.csv", index=False
    )
    print(f"\nwritten to {REPORT_DIR / 'savings_validation.csv'} and "
          f"{REPORT_DIR / 'savings_confidence_bands.csv'}")
    print(
        "\nRead this as: a recommendation is only as good as the gap behind it. "
        "Gaps under Rs.250 are close to a coin flip and should not be acted on; "
        "gaps over Rs.2,000 are near certain."
    )


if __name__ == "__main__":
    main()
