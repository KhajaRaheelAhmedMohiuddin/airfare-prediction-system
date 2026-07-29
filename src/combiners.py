"""Ensemble combiners used at the second level of the stack."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class WeightBlend:
    """Fixed convex weights over base-learner predictions (in log space)."""

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.weights


def fit_mae_weights(oof_log: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Convex weights minimising mean absolute *rupee* error.

    The base learners predict log1p(price); ridge stacking minimises squared error
    on that log scale, which quietly optimises a different objective than the
    percentage accuracy the system reports. This solves for the objective directly.
    """
    n_models = oof_log.shape[1]

    def objective(w: np.ndarray) -> float:
        return float(np.mean(np.abs(np.expm1(oof_log @ w) - y_true)))

    result = minimize(
        objective,
        x0=np.full(n_models, 1.0 / n_models),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 400, "ftol": 1e-8},
    )
    weights = result.x if result.success else np.full(n_models, 1.0 / n_models)
    return weights / weights.sum()
