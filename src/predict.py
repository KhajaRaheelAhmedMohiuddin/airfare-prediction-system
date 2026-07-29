"""Inference API for the trained airfare model.

Programmatic use:
    from predict import FarePredictor
    p = FarePredictor()
    p.predict_one(airline="IndiGo", source="Delhi", destination="Cochin",
                  date_of_journey="15/06/2019", dep_time="09:25",
                  arrival_time="04:25 16 Jun", duration="19h",
                  total_stops="2 stops", route="DEL -> LKO -> BOM -> COK")

Command line:
    python src/predict.py                 # score data/raw/Test_set.xlsx
    python src/predict.py path/to/file.xlsx
"""

from __future__ import annotations

import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from config import (
    MODEL_DIR,
    OUTPUT_DIR,
    TARGET,
    TEST_FILE,
    TRAIN_DATE_MAX,
    TRAIN_DATE_MIN,
)
from data_prep import clean, prepare
from features import add_group_statistics, build_features

BUNDLE_PATH = MODEL_DIR / "airfare_model.joblib"


class FarePredictor:
    """Loads the saved ensemble and scores new itineraries."""

    def __init__(self, bundle_path=BUNDLE_PATH) -> None:
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"No trained model at {bundle_path}. Run `python src/train.py` first."
            )
        self.bundle = joblib.load(bundle_path)
        if "models" in self.bundle:  # single-file bundles from older runs
            self.models = self.bundle["models"]
        else:
            self.models = {
                name: joblib.load(bundle_path.parent / filename)
                for name, filename in self.bundle["model_files"].items()
            }
        self.stack = self.bundle["stack"]
        self.model_names = self.bundle["model_names"]
        self.label_encoder = self.bundle["label_encoder"]
        self.target_encoder = self.bundle["target_encoder"]
        self.calibrator = self.bundle.get("calibrator")
        self.date_range = self.bundle.get(
            "date_range", (TRAIN_DATE_MIN, TRAIN_DATE_MAX)
        )
        # Market-level aggregates (route frequency, duration vs route median) were
        # computed over train+test at fit time; reproduce that same population here
        # so a single incoming row is measured against the same market.
        train_raw, test_raw = prepare(drop_duplicates=True)
        self._reference = build_features(
            pd.concat([train_raw.drop(columns=[TARGET]), test_raw], ignore_index=True)
        )

    # ------------------------------------------------------------------ core
    def _matrix(self, raw: pd.DataFrame) -> np.ndarray:
        feats = build_features(clean(raw))
        feats, _ = add_group_statistics(feats, self._reference)
        te_cols = feats[self.bundle["target_encode_cols"]].astype(str)
        encoded = self.label_encoder.transform(feats)[self.bundle["base_features"]]
        te = self.target_encoder.transform(te_cols)
        return np.hstack([encoded.to_numpy(float), te.to_numpy(float)])

    def in_training_range(self, raw: pd.DataFrame) -> np.ndarray:
        """True where the journey date falls inside the window the model was fitted on."""
        journey = pd.to_datetime(raw["Date_of_Journey"], format="%d/%m/%Y")
        lo, hi = (pd.Timestamp(d) for d in self.date_range)
        return ((journey >= lo) & (journey <= hi)).to_numpy()

    def _check_dates(self, raw: pd.DataFrame) -> np.ndarray:
        in_range = self.in_training_range(raw)
        if not in_range.all():
            lo, hi = self.date_range
            warnings.warn(
                f"{(~in_range).sum()} of {len(raw)} journey dates fall outside the "
                f"training window ({lo} to {hi}). Fares for those rows are "
                "extrapolation and should not be trusted.",
                RuntimeWarning,
                stacklevel=3,
            )
        return in_range

    def _stacked_log(self, raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self._matrix(raw)
        base = np.column_stack([self.models[n].predict(X) for n in self.model_names])
        stacked = self.stack.predict(base)
        if self.calibrator is not None:
            # Monotone correction for log-target shrinkage on expensive fares. It
            # cannot reorder itineraries, so recommendations are unaffected.
            stacked = self.calibrator.predict(stacked)
        return stacked, base

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        self._check_dates(raw)
        stacked, _ = self._stacked_log(raw)
        return np.clip(np.expm1(stacked), 1000, 120000)

    def predict_with_spread(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Point estimate plus the disagreement across base learners.

        The spread is a cheap, useful uncertainty proxy: when the five learners
        disagree the itinerary is unusual and the quoted fare is less reliable.
        """
        in_range = self._check_dates(raw)
        stacked, base = self._stacked_log(raw)
        members = np.expm1(base)
        return pd.DataFrame(
            {
                "predicted_price": np.clip(np.expm1(stacked), 1000, 120000),
                "low_estimate": members.min(axis=1),
                "high_estimate": members.max(axis=1),
                "model_spread": members.std(axis=1),
                "in_training_range": in_range,
            }
        )

    # ------------------------------------------------------------- one flight
    def predict_one(
        self,
        airline: str,
        source: str,
        destination: str,
        date_of_journey: str,
        dep_time: str,
        arrival_time: str,
        duration: str,
        total_stops: str,
        route: str | None = None,
        additional_info: str = "No info",
    ) -> float:
        row = pd.DataFrame(
            [
                {
                    "Airline": airline,
                    "Date_of_Journey": date_of_journey,
                    "Source": source,
                    "Destination": destination,
                    "Route": route or f"{source} -> {destination}",
                    "Dep_Time": dep_time,
                    "Arrival_Time": arrival_time,
                    "Duration": duration,
                    "Total_Stops": total_stops,
                    "Additional_Info": additional_info,
                }
            ]
        )
        return float(self.predict(row)[0])


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else TEST_FILE
    raw = pd.read_excel(path)
    predictor = FarePredictor()
    preds = predictor.predict(raw)

    out = OUTPUT_DIR / "predictions.xlsx"
    pd.DataFrame({TARGET: np.round(preds, 2)}).to_excel(out, index=False)
    print(f"scored {len(preds):,} itineraries from {path}")
    print(f"mean Rs.{preds.mean():,.0f}   median Rs.{np.median(preds):,.0f}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
