"""Central configuration for the Airfare Prediction System."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"


def _find(filename: str) -> Path:
    """Locate a source workbook, preferring the project root, then data/raw."""
    for candidate in (ROOT / filename, ROOT / "data" / "raw" / filename):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{filename} not found in {ROOT} or {ROOT / 'data' / 'raw'}"
    )


TRAIN_FILE = _find("Data_Train.xlsx")
TEST_FILE = _find("Test_set.xlsx")
SAMPLE_SUBMISSION_FILE = _find("Sample_submission.xlsx")

TARGET = "Price"
SEED = 42
N_FOLDS = 10

# The data is a snapshot of one season. Journey dates outside this window are
# extrapolation, and callers are warned rather than silently given a number.
TRAIN_DATE_MIN = "2019-03-01"
TRAIN_DATE_MAX = "2019-06-27"

for _d in (MODEL_DIR, OUTPUT_DIR, FIGURE_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Columns fed to the models as categorical (label/target encoded downstream).
CATEGORICAL_FEATURES = [
    "Airline",
    "Source",
    "Destination",
    "Additional_Info",
    "Route",
    "Route_Leg_1",
    "Route_Leg_2",
    "Route_Leg_3",
    "Route_Leg_4",
    "Journey_Pair",
    "Airline_Route",
    "Airline_Stops",
    "Dep_Slot",
    "Arr_Slot",
]

# Pairs used for out-of-fold target encoding (high-cardinality interactions).
TARGET_ENCODE_COLS = [
    "Airline",
    "Route",
    "Airline_Route",
    "Journey_Pair",
    "Airline_Stops",
    "Route_Leg_2",
]

# Smoothing strength per key. Anything much finer than these keys was tried and
# rejected: fine keys are empty out-of-fold but populated at inference, and the
# resulting train/serve mismatch cost accuracy on every base learner.
TARGET_ENCODE_SMOOTHING = {
    "Airline": 20.0,
    "Route": 20.0,
    "Airline_Route": 20.0,
    "Journey_Pair": 20.0,
    "Airline_Stops": 20.0,
    "Route_Leg_2": 20.0,
}
