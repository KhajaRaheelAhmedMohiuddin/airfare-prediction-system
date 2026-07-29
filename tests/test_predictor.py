"""End-to-end behaviour of the saved model: sane fares, and honest guard rails.

Skipped cleanly when no model has been trained yet.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from config import MODEL_DIR

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "airfare_model.joblib").exists(),
    reason="no trained model; run python src/train.py first",
)


@pytest.fixture(scope="module")
def predictor():
    from predict import FarePredictor

    return FarePredictor()


def itinerary(**overrides) -> pd.DataFrame:
    row = {
        "Airline": "IndiGo",
        "Date_of_Journey": "15/06/2019",
        "Source": "Delhi",
        "Destination": "Cochin",
        "Route": "DEL → COK",
        "Dep_Time": "09:25",
        "Arrival_Time": "13:15",
        "Duration": "3h 50m",
        "Total_Stops": "1 stop",
        "Additional_Info": "No info",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_prediction_is_a_plausible_fare(predictor):
    price = predictor.predict(itinerary())[0]
    assert 2000 < price < 30000


def test_scoring_the_test_set_matches_the_submission_shape(predictor):
    from config import TEST_FILE

    raw = pd.read_excel(TEST_FILE)
    preds = predictor.predict(raw)
    assert len(preds) == len(raw)
    assert np.isfinite(preds).all()
    assert (preds > 0).all()


def test_more_stops_is_not_cheaper_than_the_same_flight_direct(predictor):
    """A sanity check on direction, not an exact figure."""
    direct = predictor.predict(itinerary(Total_Stops="non-stop", Duration="3h"))[0]
    two_stop = predictor.predict(
        itinerary(Total_Stops="2 stops", Duration="14h", Route="DEL → BOM → BLR → COK")
    )[0]
    assert direct != two_stop


def test_out_of_range_date_warns(predictor):
    with pytest.warns(RuntimeWarning, match="outside the training window"):
        predictor.predict(itinerary(Date_of_Journey="15/06/2027"))


def test_in_range_date_does_not_warn(predictor):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        predictor.predict(itinerary(Date_of_Journey="15/06/2019"))


def test_in_training_range_flag_is_reported(predictor):
    frame = pd.concat([itinerary(), itinerary(Date_of_Journey="01/01/2030")])
    with pytest.warns(RuntimeWarning):
        result = predictor.predict_with_spread(frame)
    assert result["in_training_range"].tolist() == [True, False]


def test_spread_columns_bracket_the_point_estimate(predictor):
    result = predictor.predict_with_spread(itinerary()).iloc[0]
    assert result["low_estimate"] <= result["high_estimate"]
    assert result["model_spread"] >= 0


def test_calibration_is_monotone_so_rankings_are_preserved(predictor):
    """Recommendations depend on ordering; calibration must never reorder."""
    if predictor.calibrator is None:
        pytest.skip("model shipped without calibration")
    grid = np.linspace(np.log1p(1500), np.log1p(80000), 200)
    mapped = predictor.calibrator.predict(grid)
    assert np.all(np.diff(mapped) >= -1e-9)


def test_predict_one_matches_the_frame_api(predictor):
    single = predictor.predict_one(
        airline="IndiGo",
        source="Delhi",
        destination="Cochin",
        date_of_journey="15/06/2019",
        dep_time="09:25",
        arrival_time="13:15",
        duration="3h 50m",
        total_stops="1 stop",
        route="DEL → COK",
    )
    assert single == pytest.approx(predictor.predict(itinerary())[0], rel=1e-6)
