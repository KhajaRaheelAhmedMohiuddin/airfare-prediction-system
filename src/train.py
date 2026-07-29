"""End-to-end training pipeline: features -> 10-fold CV -> stacked ensemble.

Run with:  python src/train.py
Artifacts land in models/ and outputs/.
"""

from __future__ import annotations

import json
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from config import (
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    N_FOLDS,
    OUTPUT_DIR,
    REPORT_DIR,
    SEED,
    TARGET,
    TARGET_ENCODE_COLS,
    TARGET_ENCODE_SMOOTHING,
    TRAIN_DATE_MAX,
    TRAIN_DATE_MIN,
)
from combiners import WeightBlend, fit_mae_weights
from data_prep import prepare, train_target
from encoders import LabelEncoder, OutOfFoldTargetEncoder
from features import add_group_statistics, build_features
from metrics import evaluate, evaluate_by_fare_band, format_metrics
from models import build_models, fit_model

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    t0 = time.time()
    log("=" * 78)
    log("AIRFARE PREDICTION SYSTEM - training pipeline")
    log("=" * 78)

    # ------------------------------------------------------------------ data
    train_raw, test_raw = prepare(drop_duplicates=True)
    y = train_target(train_raw).to_numpy()
    y_log = np.log1p(y)
    log(f"train rows: {len(train_raw):,}   test rows: {len(test_raw):,}")

    X_train = build_features(train_raw)
    X_test = build_features(test_raw)
    X_train, X_test = add_group_statistics(X_train, X_test)
    log(f"engineered features: {X_train.shape[1]}")

    # Raw string copies of the categorical columns, kept for target encoding.
    te_train = X_train[TARGET_ENCODE_COLS].astype(str).reset_index(drop=True)
    te_test = X_test[TARGET_ENCODE_COLS].astype(str).reset_index(drop=True)

    label_encoder = LabelEncoder(CATEGORICAL_FEATURES).fit([X_train, X_test])
    X_train = label_encoder.transform(X_train).reset_index(drop=True)
    X_test = label_encoder.transform(X_test).reset_index(drop=True)

    base_features = list(X_train.columns)
    te_features = [f"TE_{c}" for c in TARGET_ENCODE_COLS]
    feature_names = base_features + te_features
    log(f"total model inputs: {len(feature_names)}")

    # -------------------------------------------------------------- CV setup
    model_names = list(build_models().keys())
    oof = {name: np.zeros(len(X_train)) for name in model_names}
    test_fold_preds = {name: [] for name in model_names}
    best_iters = {name: [] for name in model_names}

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    log("-" * 78)
    for fold, (idx_tr, idx_va) in enumerate(folds.split(X_train), start=1):
        fold_t = time.time()

        encoder = OutOfFoldTargetEncoder(
            TARGET_ENCODE_COLS, smoothing=TARGET_ENCODE_SMOOTHING, seed=SEED + fold
        )
        te_tr = encoder.fit_transform(te_train.iloc[idx_tr].reset_index(drop=True),
                                      y_log[idx_tr])
        te_va = encoder.transform(te_train.iloc[idx_va].reset_index(drop=True))
        te_te = encoder.transform(te_test)

        X_tr = np.hstack([X_train.iloc[idx_tr].to_numpy(float), te_tr.to_numpy(float)])
        X_va = np.hstack([X_train.iloc[idx_va].to_numpy(float), te_va.to_numpy(float)])
        X_te = np.hstack([X_test.to_numpy(float), te_te.to_numpy(float)])
        y_tr, y_va = y_log[idx_tr], y_log[idx_va]

        # Inner split purely for early stopping - the fold's own validation rows
        # are never shown to the boosters.
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_tr, y_tr, test_size=0.1, random_state=SEED + fold
        )

        for name, model in build_models(seed=SEED + fold).items():
            model = fit_model(name, model, X_fit, y_fit, X_es, y_es)
            oof[name][idx_va] = model.predict(X_va)
            test_fold_preds[name].append(model.predict(X_te))
            for attr in ("best_iteration_", "best_iteration"):
                if getattr(model, attr, None):
                    best_iters[name].append(int(getattr(model, attr)))
                    break
            if name == "catboost":
                best_iters[name].append(int(model.get_best_iteration() or 0))

        fold_rmse = np.sqrt(
            np.mean(
                (np.expm1(np.mean([oof[n][idx_va] for n in model_names], axis=0))
                 - y[idx_va]) ** 2
            )
        )
        log(f"fold {fold:2d}/{N_FOLDS}  blend RMSE={fold_rmse:8.2f}  "
            f"({time.time() - fold_t:.1f}s)")

    # ------------------------------------------------------- per-model scores
    log("-" * 78)
    log("Out-of-fold performance (10-fold CV, rupees):")
    results = {}
    for name in model_names:
        preds = np.expm1(oof[name])
        results[name] = evaluate(y, preds)
        log("  " + format_metrics(name, results[name]))

    # ------------------------------------------------------------- stacking
    oof_matrix = np.column_stack([oof[n] for n in model_names])
    test_matrix = np.column_stack(
        [np.mean(test_fold_preds[n], axis=0) for n in model_names]
    )

    cv2 = KFold(N_FOLDS, shuffle=True, random_state=SEED)

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 25))
    ridge_oof = cross_val_predict(ridge, oof_matrix, y_log, cv=cv2)

    # A second combiner that optimises rupee-space MAE under a simplex constraint.
    # Ridge minimises squared error in log space, which is not the same objective as
    # the percentage accuracy the business actually reports on.
    blend_oof = np.zeros(len(y))
    for idx_tr, idx_va in cv2.split(oof_matrix):
        w = fit_mae_weights(oof_matrix[idx_tr], y[idx_tr])
        blend_oof[idx_va] = oof_matrix[idx_va] @ w

    results["simple_average"] = evaluate(y, np.expm1(oof_matrix.mean(axis=1)))
    results["ridge_stack"] = evaluate(y, np.expm1(ridge_oof))
    results["mae_blend"] = evaluate(y, np.expm1(blend_oof))
    for key in ("simple_average", "ridge_stack", "mae_blend"):
        log("  " + format_metrics(key, results[key]))

    ridge.fit(oof_matrix, y_log)
    mae_w = fit_mae_weights(oof_matrix, y)

    # Deploy whichever combiner generalises better on out-of-fold data.
    use_ridge = results["ridge_stack"]["R2"] >= results["mae_blend"]["R2"]
    stack = ridge if use_ridge else WeightBlend(mae_w)
    results["stacked_ensemble"] = results["ridge_stack" if use_ridge else "mae_blend"]
    weights = (
        dict(zip(model_names, np.round(ridge.coef_, 4).tolist()))
        if use_ridge
        else dict(zip(model_names, np.round(mae_w, 4).tolist()))
    )
    log(f"combiner: {'ridge_stack' if use_ridge else 'mae_blend'}  weights={weights}")

    # ----------------------------------------------------- high-fare calibration
    # Training on log1p(price) shrinks expensive fares toward the dense middle of the
    # distribution. An isotonic map fitted on out-of-fold predictions is the obvious
    # correction, and it is monotone so it cannot reorder itineraries.
    #
    # Measured over five CV seeds, it is not worth shipping: it reduces the signed
    # bias on fares above Rs.25,000 by a stable +2.3pp, but leaves the *error* on
    # those fares unchanged (high-band MAPE moves +0.05pp, sd 0.15) while costing
    # -0.0015 R2 overall. It relabels the bias without predicting the tail any
    # better. The gate below therefore normally rejects it; it is kept, fitted and
    # reported so the trade-off stays visible rather than becoming folklore.
    raw_oof = ridge_oof if use_ridge else blend_oof
    cal_oof = np.zeros(len(y))
    for idx_tr, idx_va in cv2.split(raw_oof.reshape(-1, 1)):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_oof[idx_tr], y_log[idx_tr])
        cal_oof[idx_va] = iso.predict(raw_oof[idx_va])
    results["calibrated"] = evaluate(y, np.expm1(cal_oof))
    log("  " + format_metrics("calibrated", results["calibrated"]))

    use_calibration = results["calibrated"]["R2"] >= results["stacked_ensemble"]["R2"]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_oof, y_log)
    if not use_calibration:
        calibrator = None
        log("  calibration did not improve out-of-fold R2; shipping uncalibrated")
    else:
        results["stacked_ensemble"] = results["calibrated"]

    best_oof = cal_oof if use_calibration else raw_oof
    bands = evaluate_by_fare_band(y, np.expm1(best_oof))
    for band, stats in bands.items():
        log(f"  fare band {band:<5} n={stats['n']:<6} MAPE={stats['MAPE']:5.2f}%  "
            f"signed={stats['Signed_PE']:+6.2f}%  "
            f"actual={stats['Mean_Actual']:,.0f} predicted={stats['Mean_Predicted']:,.0f}")
    raw_bands = evaluate_by_fare_band(y, np.expm1(raw_oof))
    results["fare_bands"] = bands
    results["fare_bands_uncalibrated"] = raw_bands

    best_name = max(
        (k for k, v in results.items() if isinstance(v, dict) and "R2" in v),
        key=lambda k: results[k]["R2"],
    )
    log(f"best model: {best_name}  (R2={results[best_name]['R2']:.4f}, "
        f"accuracy={results[best_name]['Accuracy_100_minus_MAPE']:.2f}%)")

    # ------------------------------------------- refit on 100% of train data
    log("-" * 78)
    log("Refitting base learners on the full training set for deployment...")
    final_encoder = OutOfFoldTargetEncoder(
        TARGET_ENCODE_COLS, smoothing=TARGET_ENCODE_SMOOTHING, seed=SEED
    )
    te_full = final_encoder.fit_transform(te_train, y_log)
    X_full = np.hstack([X_train.to_numpy(float), te_full.to_numpy(float)])

    final_models = {}
    for name, model in build_models(seed=SEED).items():
        if best_iters[name]:
            n_est = max(int(np.mean(best_iters[name]) * 1.1), 100)
            if name == "lightgbm":
                model.set_params(n_estimators=n_est)
            elif name == "xgboost":
                model.set_params(n_estimators=n_est, early_stopping_rounds=None)
            elif name == "catboost":
                model.set_params(iterations=n_est, early_stopping_rounds=None)
        model.fit(X_full, y_log)
        final_models[name] = model
    log("  done.")

    # ------------------------------------------------------------ artifacts
    # The two forests are ~90% of the artifact and a single combined file exceeds the
    # 100 MB that most git hosts refuse. Each learner gets its own file instead;
    # shrinking them to fit was measured and costs real accuracy (see README).
    for name, model in final_models.items():
        joblib.dump(model, MODEL_DIR / f"base_{name}.joblib", compress=6)

    bundle = {
        "model_files": {n: f"base_{n}.joblib" for n in final_models},
        "stack": stack,
        "model_names": model_names,
        "label_encoder": label_encoder,
        "target_encoder": final_encoder,
        "feature_names": feature_names,
        "base_features": base_features,
        "target_encode_cols": TARGET_ENCODE_COLS,
        "calibrator": calibrator,
        "date_range": (TRAIN_DATE_MIN, TRAIN_DATE_MAX),
        "metrics": results,
    }
    path = MODEL_DIR / "airfare_model.joblib"
    joblib.dump(bundle, path, compress=6)
    sizes = {
        f.name: f.stat().st_size / 1e6 for f in MODEL_DIR.glob("*.joblib")
    }
    log(f"saved model files   -> {MODEL_DIR}  total {sum(sizes.values()):.0f} MB, "
        f"largest {max(sizes.values()):.0f} MB")

    # Test-set predictions come from the fold-averaged base models fed through
    # the fitted stack - more robust than any single full-data refit.
    stacked_log = stack.predict(test_matrix)
    if calibrator is not None:
        stacked_log = calibrator.predict(stacked_log)
    test_pred = np.expm1(stacked_log)
    test_pred = np.clip(test_pred, y.min() * 0.6, y.max() * 1.4)
    submission = pd.DataFrame({TARGET: np.round(test_pred, 2)})
    submission.to_excel(OUTPUT_DIR / "submission.xlsx", index=False)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    log(f"saved predictions   -> {OUTPUT_DIR / 'submission.xlsx'} "
        f"({len(submission):,} rows, mean Rs.{test_pred.mean():,.0f})")

    np.save(MODEL_DIR / "oof_predictions.npy", oof_matrix)
    flat = {k: v for k, v in results.items() if "R2" in v}
    pd.DataFrame(flat).T.to_csv(REPORT_DIR / "cv_metrics.csv")
    pd.DataFrame(results["fare_bands"]).T.to_csv(REPORT_DIR / "fare_band_metrics.csv")
    with open(REPORT_DIR / "cv_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "metrics": results,
                "stack_weights": weights,
                "n_features": len(feature_names),
                "calibrated": bool(use_calibration),
            },
            fh,
            indent=2,
        )
    pd.DataFrame({"y_true": y, "y_pred": np.expm1(best_oof)}).to_csv(
        REPORT_DIR / "oof_stacked.csv", index=False
    )

    # LightGBM gain importance from the full-data refit.
    lgb = final_models["lightgbm"]
    imp = pd.DataFrame(
        {"feature": feature_names, "gain": lgb.booster_.feature_importance("gain")}
    ).sort_values("gain", ascending=False)
    imp.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    log("top 15 features by gain: " + ", ".join(imp.head(15)["feature"]))

    log("-" * 78)
    log(f"total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
