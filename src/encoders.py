"""Categorical encoders used by the modelling pipeline.

`OutOfFoldTargetEncoder` is the important one: it replaces a high-cardinality
category with a smoothed mean of the (log) target, but the values it hands back
for the rows it was fitted on come from inner folds, so no row ever sees its own
target. This is what keeps the cross-validation score honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class OutOfFoldTargetEncoder:
    def __init__(
        self,
        columns: list[str],
        n_inner_folds: int = 5,
        smoothing: float | dict[str, float] = 20.0,
        seed: int = 42,
        with_counts: bool = False,
    ) -> None:
        self.columns = columns
        self.n_inner_folds = n_inner_folds
        self.smoothing = smoothing
        self.seed = seed
        # Emitting group sizes alongside the encoded means measured slightly worse
        # in cross-validation, so it is off by default.
        self.with_counts = with_counts
        self.maps_: dict[str, pd.Series] = {}
        self.count_maps_: dict[str, pd.Series] = {}
        self.prior_: float = 0.0

    def _alpha(self, col: str) -> float:
        if isinstance(self.smoothing, dict):
            return float(self.smoothing.get(col, 20.0))
        return float(self.smoothing)

    def _stats(self, keys: pd.Series, y: np.ndarray) -> pd.DataFrame:
        return (
            pd.DataFrame({"key": keys.to_numpy(), "y": y})
            .groupby("key")["y"]
            .agg(["mean", "count"])
        )

    def _smoothed_map(self, col: str, keys: pd.Series, y: np.ndarray) -> pd.Series:
        stats = self._stats(keys, y)
        weight = stats["count"] / (stats["count"] + self._alpha(col))
        return weight * stats["mean"] + (1 - weight) * self.prior_

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        self.prior_ = float(np.mean(y))
        encoded = pd.DataFrame(index=X.index)
        folds = KFold(n_splits=self.n_inner_folds, shuffle=True, random_state=self.seed)

        for col in self.columns:
            oof = np.full(len(X), self.prior_, dtype=float)
            oof_count = np.zeros(len(X), dtype=float)
            for inner_train, inner_valid in folds.split(X):
                # The fallback prior is computed from the inner-training rows only.
                # Using the global mean here would let every row leak a 1/n share of
                # its own target into its own encoded value - small, but real.
                fold_prior = float(np.mean(y[inner_train]))
                stats = self._stats(X[col].iloc[inner_train], y[inner_train])
                weight = stats["count"] / (stats["count"] + self._alpha(col))
                mapping = weight * stats["mean"] + (1 - weight) * fold_prior
                keys = X[col].iloc[inner_valid]
                oof[inner_valid] = keys.map(mapping).fillna(fold_prior).to_numpy()
                oof_count[inner_valid] = keys.map(stats["count"]).fillna(0).to_numpy()
            encoded[f"TE_{col}"] = oof
            if self.with_counts:
                # Group size tells the trees how much to trust the encoded mean.
                encoded[f"TEN_{col}"] = oof_count

            # Full-data mapping is what unseen rows (valid fold / test) will use.
            full = self._stats(X[col], y)
            w = full["count"] / (full["count"] + self._alpha(col))
            self.maps_[col] = w * full["mean"] + (1 - w) * self.prior_
            self.count_maps_[col] = full["count"]

        return encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.DataFrame(index=X.index)
        for col in self.columns:
            encoded[f"TE_{col}"] = (
                X[col].map(self.maps_[col]).fillna(self.prior_).to_numpy()
            )
            if self.with_counts:
                encoded[f"TEN_{col}"] = (
                    X[col].map(self.count_maps_[col]).fillna(0).to_numpy()
                )
        return encoded


class LabelEncoder:
    """Deterministic string -> integer codes, with a bucket for unseen values."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.maps_: dict[str, dict[str, int]] = {}

    def fit(self, frames: list[pd.DataFrame]) -> "LabelEncoder":
        for col in self.columns:
            values = pd.concat([f[col].astype(str) for f in frames], ignore_index=True)
            categories = sorted(values.unique())
            self.maps_[col] = {v: i for i, v in enumerate(categories)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            out[col] = out[col].astype(str).map(self.maps_[col]).fillna(-1).astype(int)
        return out
