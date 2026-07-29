"""Loading and cleaning of the raw airfare workbooks."""

from __future__ import annotations

import pandas as pd

from config import SAMPLE_SUBMISSION_FILE, TARGET, TEST_FILE, TRAIN_FILE

# Destination values that refer to the same airport city.
CITY_ALIASES = {"New Delhi": "Delhi"}

# Additional_Info arrives with inconsistent casing and a couple of near-duplicates.
INFO_ALIASES = {
    "No Info": "No info",
    "No info": "No info",
}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three workbooks exactly as delivered."""
    train = pd.read_excel(TRAIN_FILE)
    test = pd.read_excel(TEST_FILE)
    sample = pd.read_excel(SAMPLE_SUBMISSION_FILE)
    return train, test, sample


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise text columns and fill the handful of missing values."""
    out = df.copy()

    for col in ("Airline", "Source", "Destination", "Route", "Additional_Info"):
        out[col] = out[col].astype("string").str.strip()

    out["Additional_Info"] = out["Additional_Info"].replace(INFO_ALIASES)
    out["Destination_City"] = out["Destination"].replace(CITY_ALIASES)
    out["Source_City"] = out["Source"].replace(CITY_ALIASES)

    # One record in train has a null Route/Total_Stops; recover it from the city pair.
    out["Total_Stops"] = out["Total_Stops"].fillna("1 stop")
    route_fallback = out["Source"].astype(str) + " -> " + out["Destination"].astype(str)
    out["Route"] = out["Route"].fillna(pd.Series(route_fallback, index=out.index))

    return out


def prepare(drop_duplicates: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned train/test frames ready for feature engineering."""
    train, test, _ = load_raw()

    if drop_duplicates:
        # 220 rows are byte-identical repeats (same itinerary *and* same fare).
        train = train.drop_duplicates().reset_index(drop=True)

    return clean(train), clean(test)


def train_target(train: pd.DataFrame) -> pd.Series:
    return train[TARGET].astype(float)
