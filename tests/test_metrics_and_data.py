"""Metric definitions and the data-cleaning contract."""

import numpy as np
import pandas as pd
import pytest

from data_prep import clean, load_raw, prepare
from metrics import evaluate, evaluate_by_fare_band


def test_perfect_predictions_score_perfectly():
    y = np.array([1000.0, 5000.0, 20000.0])
    m = evaluate(y, y.copy())
    assert m["R2"] == pytest.approx(1.0)
    assert m["MAE"] == pytest.approx(0.0)
    assert m["MAPE"] == pytest.approx(0.0)
    assert m["Accuracy_100_minus_MAPE"] == pytest.approx(100.0)
    assert m["Within_5pct"] == pytest.approx(100.0)


def test_accuracy_is_the_complement_of_mape():
    y = np.array([100.0, 200.0])
    pred = np.array([110.0, 180.0])
    m = evaluate(y, pred)
    assert m["MAPE"] == pytest.approx(10.0)
    assert m["Accuracy_100_minus_MAPE"] == pytest.approx(90.0)


def test_within_thresholds_are_ordered():
    rng = np.random.default_rng(0)
    y = rng.uniform(2000, 30000, 500)
    pred = y * rng.normal(1.0, 0.1, 500)
    m = evaluate(y, pred)
    assert m["Within_5pct"] <= m["Within_10pct"] <= m["Within_20pct"]


def test_fare_band_signed_error_detects_under_prediction():
    """Under-predicting expensive fares must show up as a negative signed error."""
    y = np.array([5000.0, 6000.0, 40000.0, 50000.0])
    pred = np.array([5000.0, 6000.0, 30000.0, 35000.0])
    bands = evaluate_by_fare_band(y, pred, threshold=25000)
    assert bands["high"]["n"] == 2
    assert bands["high"]["Signed_PE"] < -10
    assert bands["low"]["Signed_PE"] == pytest.approx(0.0)


def test_duplicate_rows_are_dropped():
    raw_train, _, _ = load_raw()
    train, _ = prepare(drop_duplicates=True)
    assert len(train) < len(raw_train)
    assert not train.duplicated().any()


def test_cleaning_fills_the_missing_route_and_stops():
    raw_train, _, _ = load_raw()
    assert raw_train["Route"].isnull().any()
    cleaned = clean(raw_train)
    assert not cleaned["Route"].isnull().any()
    assert not cleaned["Total_Stops"].isnull().any()


def test_additional_info_casing_is_normalised():
    cleaned = clean(
        pd.DataFrame(
            [
                {
                    "Airline": "IndiGo",
                    "Source": "Delhi",
                    "Destination": "Cochin",
                    "Route": "DEL → COK",
                    "Additional_Info": "No Info",
                    "Total_Stops": "1 stop",
                }
            ]
        )
    )
    assert cleaned["Additional_Info"].iloc[0] == "No info"


def test_test_set_row_count_matches_the_sample_submission():
    _, test, sample = load_raw()
    assert len(test) == len(sample)
