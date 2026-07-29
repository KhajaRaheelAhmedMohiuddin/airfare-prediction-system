"""Base learners for the stacked airfare ensemble."""

from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from config import SEED


def build_models(seed: int = SEED) -> dict:
    """Fresh, unfitted base learners keyed by name."""
    return {
        "lightgbm": LGBMRegressor(
            n_estimators=4000,
            learning_rate=0.02,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=8,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=4000,
            learning_rate=0.02,
            max_depth=9,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=200,
            verbosity=0,
        ),
        "catboost": CatBoostRegressor(
            iterations=4000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3.0,
            random_seed=seed,
            loss_function="RMSE",
            early_stopping_rounds=200,
            verbose=0,
            allow_writing_files=False,
        ),
        # Leaf size and tree count are held down deliberately: fully grown 700-tree
        # forests produced a 250 MB deployment bundle for a fraction of a point of
        # accuracy. These settings keep the artifact small enough to move around.
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_features=0.5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            max_features=0.7,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
    }


def fit_model(name: str, model, X_fit, y_fit, X_es, y_es):
    """Fit one base learner, using an inner split for early stopping where supported.

    The early-stopping split is carved out of the *training* part of the fold, never
    the fold's validation rows, so the cross-validated score stays unbiased.
    """
    if name == "lightgbm":
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_es, y_es)],
            eval_metric="rmse",
            callbacks=[early_stopping(200, verbose=False), log_evaluation(0)],
        )
    elif name == "xgboost":
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    elif name == "catboost":
        model.fit(X_fit, y_fit, eval_set=(X_es, y_es), verbose=0)
    else:
        # Forests do not early stop; give them every row available.
        model.fit(np.vstack([X_fit, X_es]), np.concatenate([y_fit, y_es]))
    return model
