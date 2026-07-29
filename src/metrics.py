"""Regression metrics, including the business-facing accuracy definitions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    rmsle = float(
        np.sqrt(np.mean((np.log1p(np.maximum(y_pred, 0)) - np.log1p(y_true)) ** 2))
    )

    def within(pct: float) -> float:
        return float(np.mean(np.abs(y_true - y_pred) / y_true <= pct) * 100)

    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape,
        "Accuracy_100_minus_MAPE": 100.0 - mape,
        "RMSLE": rmsle,
        "Within_5pct": within(0.05),
        "Within_10pct": within(0.10),
        "Within_20pct": within(0.20),
    }


HIGH_FARE_THRESHOLD = 25000.0


def evaluate_by_fare_band(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = HIGH_FARE_THRESHOLD
) -> dict[str, dict[str, float]]:
    """Split the score by fare band and report *signed* error.

    The headline metrics are dominated by ordinary tickets. Expensive fares are rare
    and the model shrinks them toward the middle of the distribution, so the signed
    bias in the top band is the number that reveals it.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    bands = {"low": y_true <= threshold, "high": y_true > threshold}

    out: dict[str, dict[str, float]] = {}
    for name, mask in bands.items():
        if not mask.any():
            continue
        yt, yp = y_true[mask], y_pred[mask]
        out[name] = {
            "n": int(mask.sum()),
            "MAPE": float(np.mean(np.abs((yt - yp) / yt)) * 100),
            "Signed_PE": float(np.mean((yp - yt) / yt) * 100),
            "Mean_Actual": float(yt.mean()),
            "Mean_Predicted": float(yp.mean()),
        }
    return out


def format_metrics(name: str, m: dict[str, float]) -> str:
    return (
        f"{name:<16} R2={m['R2']:.4f}  RMSE={m['RMSE']:8.2f}  MAE={m['MAE']:7.2f}  "
        f"MAPE={m['MAPE']:5.2f}%  Acc={m['Accuracy_100_minus_MAPE']:5.2f}%  "
        f"<=10%={m['Within_10pct']:5.2f}%"
    )
