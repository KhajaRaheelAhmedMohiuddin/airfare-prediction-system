"""Feature engineering must be correct row-by-row, and stable in shape."""

import numpy as np
import pandas as pd
import pytest

from data_prep import clean
from features import _duration_to_minutes, _time_to_minutes, build_features


def make_raw(**overrides) -> pd.DataFrame:
    row = {
        "Airline": "IndiGo",
        "Date_of_Journey": "24/03/2019",
        "Source": "Banglore",
        "Destination": "New Delhi",
        "Route": "BLR → DEL",
        "Dep_Time": "22:20",
        "Arrival_Time": "01:10 22 Mar",
        "Duration": "2h 50m",
        "Total_Stops": "non-stop",
        "Additional_Info": "No info",
    }
    row.update(overrides)
    return clean(pd.DataFrame([row]))


@pytest.mark.parametrize(
    "text,expected",
    [("2h 50m", 170), ("19h", 1140), ("5m", 5), ("21h 5m", 1265), ("0h 0m", 0)],
)
def test_duration_parsing(text, expected):
    assert _duration_to_minutes(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [("05:50", 350), ("22:20", 1340), ("01:10 22 Mar", 70), ("00:00", 0)],
)
def test_time_parsing(text, expected):
    assert _time_to_minutes(text) == expected


def test_flight_longer_than_a_day_is_flagged():
    feats = build_features(
        make_raw(Dep_Time="22:20", Arrival_Time="01:10 24 Mar", Duration="25h 30m")
    )
    assert feats["Is_Multi_Day_Flight"].iloc[0] == 1


def test_midnight_crossing_alone_is_not_a_multi_day_flight():
    """22:20 -> 01:10 next day is under 24h; the arrival clock already covers it."""
    feats = build_features(make_raw())
    assert feats["Is_Multi_Day_Flight"].iloc[0] == 0
    assert feats["Is_Red_Eye"].iloc[0] == 1


def test_same_day_flight_is_not_flagged():
    feats = build_features(make_raw(Dep_Time="05:50", Arrival_Time="13:15",
                                    Duration="7h 25m"))
    assert feats["Is_Multi_Day_Flight"].iloc[0] == 0


def test_cyclical_time_encoding_wraps_around():
    """23:50 and 00:10 are 20 minutes apart and must encode as neighbours."""
    late = build_features(make_raw(Dep_Time="23:50"))
    early = build_features(make_raw(Dep_Time="00:10"))
    distance = np.hypot(
        late["Dep_Sin"].iloc[0] - early["Dep_Sin"].iloc[0],
        late["Dep_Cos"].iloc[0] - early["Dep_Cos"].iloc[0],
    )
    assert distance < 0.1


def test_stops_map_to_ordinal_and_direct_flag():
    direct = build_features(make_raw(Total_Stops="non-stop"))
    two = build_features(make_raw(Total_Stops="2 stops"))
    assert direct["Total_Stops"].iloc[0] == 0
    assert direct["Is_Direct"].iloc[0] == 1
    assert two["Total_Stops"].iloc[0] == 2
    assert two["Is_Direct"].iloc[0] == 0


def test_route_is_split_into_legs():
    feats = build_features(make_raw(Route="CCU → IXR → BBI → BLR",
                                    Total_Stops="2 stops"))
    assert feats["Route_Hops"].iloc[0] == 4
    assert feats["Route_Leg_1"].iloc[0] == "CCU"
    assert feats["Route_Leg_2"].iloc[0] == "IXR"
    assert feats["Route_Leg_4"].iloc[0] == "BLR"


def test_new_delhi_and_delhi_are_the_same_city():
    a = build_features(make_raw(Destination="New Delhi"))
    b = build_features(make_raw(Destination="Delhi"))
    assert a["Destination"].iloc[0] == b["Destination"].iloc[0] == "Delhi"


def test_holiday_distance_is_zero_on_a_holiday():
    """21 March 2019 is Holi."""
    feats = build_features(make_raw(Date_of_Journey="21/03/2019"))
    assert feats["Days_To_Holiday"].iloc[0] == 0
    assert feats["Is_Holiday_Week"].iloc[0] == 1


def test_service_flags_read_additional_info():
    feats = build_features(make_raw(Additional_Info="In-flight meal not included"))
    assert feats["No_Meal"].iloc[0] == 1
    assert feats["No_Baggage"].iloc[0] == 0


def test_features_are_never_null_on_the_real_data():
    from data_prep import prepare

    train, test = prepare()
    for frame in (build_features(train), build_features(test)):
        nulls = frame.isnull().sum()
        assert not nulls.any(), f"null features: {list(nulls[nulls > 0].index)}"


def test_train_and_test_produce_identical_feature_columns():
    from data_prep import prepare

    train, test = prepare()
    assert list(build_features(train).columns) == list(build_features(test).columns)
