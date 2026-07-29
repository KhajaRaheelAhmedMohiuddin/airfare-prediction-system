"""Feature engineering for the Airfare Prediction System.

Every transform here is row-wise and therefore leakage-free: nothing looks at the
target. Target-statistics encoding lives in `encoders.py` and is fitted inside the
cross-validation loop instead.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

STOP_MAP = {
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4,
}

# Indian public holidays / festivals inside the Mar-Jun 2019 window covered by the
# data. Fares spike around these dates, so distance-to-holiday carries signal.
HOLIDAYS_2019 = pd.to_datetime(
    [
        "2019-03-04",  # Maha Shivaratri
        "2019-03-21",  # Holi
        "2019-04-13",  # Ram Navami
        "2019-04-17",  # Mahavir Jayanti
        "2019-04-19",  # Good Friday
        "2019-05-01",  # May Day
        "2019-05-18",  # Buddha Purnima
        "2019-06-05",  # Eid al-Fitr
    ]
)

_DURATION_RE = re.compile(r"(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?")


def _duration_to_minutes(text: str) -> int:
    match = _DURATION_RE.match(str(text).strip())
    if not match:
        return np.nan
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    return hours * 60 + minutes


def _time_to_minutes(text: str) -> int:
    """'01:10 22 Mar' and '05:50' both reduce to minutes past midnight."""
    clock = str(text).strip().split(" ")[0]
    hours, minutes = clock.split(":")
    return int(hours) * 60 + int(minutes)


def _slot(minutes: pd.Series) -> pd.Series:
    """Bucket a minutes-past-midnight series into named departure windows."""
    bins = [-1, 240, 480, 720, 960, 1200, 1440]
    labels = ["red_eye", "early_morning", "morning", "afternoon", "evening", "night"]
    return pd.cut(minutes, bins=bins, labels=labels).astype(str)


def _split_route(route: pd.Series, max_legs: int = 5) -> pd.DataFrame:
    legs = route.astype(str).str.split(r"\s*(?:→|->)\s*", regex=True)
    frame = pd.DataFrame(index=route.index)
    frame["Route_Hops"] = legs.apply(len)
    for i in range(max_legs):
        frame[f"Route_Node_{i + 1}"] = legs.apply(
            lambda x, i=i: x[i] if len(x) > i else "NONE"
        )
    return frame


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a cleaned raw frame into the full modelling feature table."""
    out = pd.DataFrame(index=df.index)

    # --- Journey date -------------------------------------------------------
    journey = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    out["Journey_Day"] = journey.dt.day
    out["Journey_Month"] = journey.dt.month
    out["Journey_DayOfWeek"] = journey.dt.dayofweek
    out["Journey_DayOfYear"] = journey.dt.dayofyear
    out["Journey_WeekOfYear"] = journey.dt.isocalendar().week.astype(int)
    out["Journey_Quarter_Day"] = journey.dt.day // 8  # coarse in-month position
    out["Is_Weekend"] = journey.dt.dayofweek.isin([5, 6]).astype(int)
    out["Is_Month_Start"] = journey.dt.is_month_start.astype(int)
    out["Is_Month_End"] = journey.dt.is_month_end.astype(int)

    # Distance (in days) to the nearest festival/public holiday.
    diffs = np.abs(
        journey.values.astype("datetime64[D]")[:, None]
        - HOLIDAYS_2019.values.astype("datetime64[D]")[None, :]
    ).astype(int)
    out["Days_To_Holiday"] = diffs.min(axis=1)
    out["Is_Holiday_Week"] = (out["Days_To_Holiday"] <= 3).astype(int)

    # --- Departure / arrival clocks ----------------------------------------
    dep_min = df["Dep_Time"].map(_time_to_minutes)
    arr_min = df["Arrival_Time"].map(_time_to_minutes)
    out["Dep_Minutes"] = dep_min
    out["Dep_Hour"] = dep_min // 60
    out["Dep_Minute"] = dep_min % 60
    out["Arr_Minutes"] = arr_min
    out["Arr_Hour"] = arr_min // 60
    out["Arr_Minute"] = arr_min % 60
    out["Dep_Slot"] = _slot(dep_min)
    out["Arr_Slot"] = _slot(arr_min)
    out["Is_Red_Eye"] = ((dep_min < 300) | (arr_min < 300)).astype(int)
    # Cyclical encodings so 23:50 and 00:10 sit next to each other.
    out["Dep_Sin"] = np.sin(2 * np.pi * dep_min / 1440)
    out["Dep_Cos"] = np.cos(2 * np.pi * dep_min / 1440)
    out["Arr_Sin"] = np.sin(2 * np.pi * arr_min / 1440)
    out["Arr_Cos"] = np.cos(2 * np.pi * arr_min / 1440)

    # --- Duration -----------------------------------------------------------
    duration = df["Duration"].map(_duration_to_minutes)
    out["Duration_Minutes"] = duration
    out["Duration_Hours"] = duration // 60
    out["Duration_Log"] = np.log1p(duration)
    # Duration in excess of the elapsed time the arrival clock implies, in whole days.
    # This is *not* a midnight-crossing flag - a 22:20 departure landing 01:10 the next
    # day scores 0, because the arrival clock already accounts for it. It fires only on
    # itineraries longer than 24 hours, which are priced differently.
    out["Is_Multi_Day_Flight"] = np.ceil(
        np.maximum(duration - ((arr_min - dep_min) % 1440), 0) / 1440
    ).astype(int)

    # --- Stops --------------------------------------------------------------
    stops = df["Total_Stops"].map(STOP_MAP).fillna(1).astype(int)
    out["Total_Stops"] = stops
    out["Is_Direct"] = (stops == 0).astype(int)
    out["Duration_Per_Stop"] = duration / (stops + 1)

    # --- Route --------------------------------------------------------------
    route_parts = _split_route(df["Route"])
    out["Route_Hops"] = route_parts["Route_Hops"]
    for i in range(1, 5):
        out[f"Route_Leg_{i}"] = route_parts[f"Route_Node_{i}"]

    # --- Raw categoricals passed through -----------------------------------
    out["Airline"] = df["Airline"].astype(str)
    out["Source"] = df["Source_City"].astype(str)
    out["Destination"] = df["Destination_City"].astype(str)
    out["Additional_Info"] = df["Additional_Info"].astype(str)
    out["Route"] = df["Route"].astype(str)

    # --- Interactions -------------------------------------------------------
    out["Journey_Pair"] = out["Source"] + "_" + out["Destination"]
    out["Airline_Route"] = out["Airline"] + "_" + out["Route"]
    out["Airline_Stops"] = out["Airline"] + "_" + stops.astype(str)

    # Note: date-specific identity keys (this flight on this date) were tried here and
    # removed. They look attractive - 60% of test rows share a flight *and* date with a
    # training row - but out-of-fold their group size is almost always 1, so during
    # training they collapse to the global prior while at inference they carry a real
    # fare. Every base learner lost accuracy to that train/serve mismatch.

    # --- Fare-relevant service flags ---------------------------------------
    airline_lower = out["Airline"].str.lower()
    out["Is_Premium_Cabin"] = airline_lower.str.contains(
        "business|premium", regex=True
    ).astype(int)
    out["Is_Low_Cost"] = airline_lower.str.contains(
        "indigo|spicejet|goair|air asia|trujet", regex=True
    ).astype(int)
    info_lower = out["Additional_Info"].str.lower()
    out["No_Meal"] = info_lower.str.contains("meal not included").astype(int)
    out["No_Baggage"] = info_lower.str.contains("no check-in baggage").astype(int)
    out["Long_Layover"] = info_lower.str.contains("long layover").astype(int)
    out["Change_Airports"] = info_lower.str.contains("change airports").astype(int)

    return out


def add_group_statistics(
    train_feats: pd.DataFrame, test_feats: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add unsupervised group aggregates computed on train+test together.

    These describe the *market* (how many itineraries exist on a route, how this
    flight's duration compares to the route median) and never touch the target,
    so pooling both splits is safe and gives more stable estimates.
    """
    train_feats = train_feats.copy()
    test_feats = test_feats.copy()
    combined = pd.concat([train_feats, test_feats], axis=0, ignore_index=True)
    n_train = len(train_feats)

    for col in ["Airline", "Route", "Journey_Pair", "Airline_Route", "Route_Leg_2"]:
        counts = combined[col].map(combined[col].value_counts())
        train_feats[f"{col}_Freq"] = counts.iloc[:n_train].to_numpy()
        test_feats[f"{col}_Freq"] = counts.iloc[n_train:].to_numpy()

    for group in ["Journey_Pair", "Airline_Route"]:
        med = combined.groupby(group)["Duration_Minutes"].transform("median")
        mn = combined.groupby(group)["Duration_Minutes"].transform("min")
        rel = combined["Duration_Minutes"] - med
        excess = combined["Duration_Minutes"] - mn
        train_feats[f"Dur_vs_{group}_Median"] = rel.iloc[:n_train].to_numpy()
        test_feats[f"Dur_vs_{group}_Median"] = rel.iloc[n_train:].to_numpy()
        train_feats[f"Dur_over_{group}_Min"] = excess.iloc[:n_train].to_numpy()
        test_feats[f"Dur_over_{group}_Min"] = excess.iloc[n_train:].to_numpy()

    # How busy is that departure date overall (demand proxy)?
    day_counts = combined["Journey_DayOfYear"].map(
        combined["Journey_DayOfYear"].value_counts()
    )
    train_feats["Day_Flight_Count"] = day_counts.iloc[:n_train].to_numpy()
    test_feats["Day_Flight_Count"] = day_counts.iloc[n_train:].to_numpy()

    return train_feats, test_feats
