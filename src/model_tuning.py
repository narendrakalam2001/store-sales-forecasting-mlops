# ============================================================
# MODEL TUNING — Store Sales Forecasting System
# ============================================================
# Three model families:
#
#   1. LightGBM  — primary tabular model
#      - RMSE on log1p(sales) as objective
#      - Early stopping on validation set
#      - Feature importance extracted post-training
#
#   2. Prophet   — per store-family trend + seasonality
#      - Multiplicative seasonality (retail EDA finding)
#      - National holiday regressors
#      - Oil price as external regressor
#
#   3. LSTM      — sequence model
#      - Input: 28-day window of scaled features
#      - Output: next-day log1p(sales)
#      - Early stopping on val loss
#
#   4. Ensemble  — weighted average (grid-search tuned)
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd

from typing import Dict, Optional, Tuple

from src.config import (
    RANDOM_STATE, N_JOBS, MODEL_DIR,
    LGBM_PARAMS, PROPHET_PARAMS,
    LSTM_SEQ_LEN, LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_PATIENCE,
    ENSEMBLE_WEIGHTS, FORECAST_HORIZON,
)
from src.metrics import rmsle

logger = logging.getLogger(__name__)


# ============================================================
# 1. LIGHTGBM
# ============================================================

def train_lightgbm(
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    X_val:         np.ndarray,
    y_val:         np.ndarray,
    feature_names: list = None,
    params:        dict = None,
) -> Tuple:
    """
    Trains LightGBM on log1p(sales) with early stopping.

    EDA rationale:
        - LightGBM handles: lag features (skewed), zero-inflation,
          categorical encodings, without scaling
        - 1000 trees with early stopping @ 100 rounds
        - RMSE on log target is equivalent to RMSLE on raw target

    Returns: (lgbm_model, val_rmsle, feature_importance_df)
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")

    run_params = {**LGBM_PARAMS, **(params or {})}

    dtrain = lgb.Dataset(
        X_train, label=y_train,
        feature_name=feature_names or "auto",
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val, label=y_val,
        reference=dtrain,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    logger.info(
        "Training LightGBM  |  train=%d  val=%d",
        len(X_train), len(X_val)
    )
    start = time.time()

    model = lgb.train(
        params          = run_params,
        train_set       = dtrain,
        num_boost_round = run_params.get("n_estimators", 1000),
        valid_sets      = [dval],
        callbacks       = callbacks,
    )

    # Evaluate
    y_log_pred = model.predict(X_val)
    y_pred     = np.expm1(y_log_pred).clip(0)
    y_true     = np.expm1(y_val).clip(0)
    val_rmsle  = rmsle(y_true, y_pred)

    logger.info(
        "LightGBM done  |  best_iter=%d  val_RMSLE=%.5f  time=%.1fs",
        model.best_iteration, val_rmsle, time.time() - start
    )

    # Feature importance
    fi_df = None
    try:
        fi_vals  = model.feature_importance(importance_type="gain")
        fi_names = feature_names if feature_names else [f"f{i}" for i in range(len(fi_vals))]
        fi_df = (
            pd.DataFrame({"feature": fi_names, "importance": fi_vals})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as e:
        logger.warning("Feature importance failed: %s", e)

    return model, val_rmsle, fi_df


# ============================================================
# 2. PROPHET — per (store, family) series
# ============================================================

def _train_single_prophet(args: tuple) -> Tuple:
    """Trains one Prophet model for a single (store_nbr, family)."""
    store_nbr, family, series_df, prophet_holidays, params = args
    key = f"{store_nbr}_{family}"

    try:
        from prophet import Prophet
    except ImportError:
        return key, None, float("inf")

    try:
        df_p = series_df[["date", "sales", "dcoilwtico"]].copy()
        df_p = df_p.rename(columns={"date": "ds"})
        df_p["y"] = np.log1p(df_p["sales"].clip(0))

        cutoff   = df_p["ds"].max() - pd.Timedelta(FORECAST_HORIZON, "D")
        df_train = df_p[df_p["ds"] <= cutoff].copy()
        df_val   = df_p[df_p["ds"] > cutoff].copy()

        if len(df_train) < 60:
            return key, None, float("inf")

        model = Prophet(
            yearly_seasonality       = params.get("yearly_seasonality", True),
            weekly_seasonality       = params.get("weekly_seasonality", True),
            daily_seasonality        = params.get("daily_seasonality",  False),
            seasonality_mode         = params.get("seasonality_mode", "multiplicative"),
            changepoint_prior_scale  = params.get("changepoint_prior_scale", 0.05),
            seasonality_prior_scale  = params.get("seasonality_prior_scale", 10.0),
            holidays                 = prophet_holidays,
        )

        model.add_regressor("dcoilwtico", standardize=True)
        model.fit(df_train[["ds", "y", "dcoilwtico"]])

        future   = df_val[["ds", "dcoilwtico"]].copy()
        forecast = model.predict(future)

        y_pred = np.expm1(forecast["yhat"].values).clip(0)
        y_true = np.expm1(df_val["y"].values).clip(0)

        return key, model, rmsle(y_true, y_pred)

    except Exception as e:
        logger.warning("Prophet failed for %s: %s", key, e)
        return key, None, float("inf")


def train_prophet_models(
    df:         pd.DataFrame,
    holidays:   pd.DataFrame,
    params:     dict = None,
    max_series: int  = None,
) -> Dict[str, object]:
    """
    Trains one Prophet per (store_nbr, family) — 54 x 33 = 1782 series.

    EDA rationale:
        - Multiplicative seasonality: Corporacion Favorita is growing chain
          (multiplicative captures percentage growth better than additive)
        - National holidays: +23% sales spike (EDA Mann-Whitney finding)
        - Oil: negative macro correlation (EDA finding: corr = -0.31)

    max_series: limit for quick testing (None = all 1782 series).

    Returns: dict of {"store_nbr_family": prophet_model}
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet not installed — skipping")
        return {}

    run_params = {**PROPHET_PARAMS, **(params or {})}

    # Build Prophet holidays DataFrame
    prophet_holidays = None
    if "is_national_holiday" in holidays.columns:
        nat = holidays[holidays["is_national_holiday"] == 1]
        if len(nat) > 0:
            prophet_holidays = pd.DataFrame({
                "holiday":      "national_holiday",
                "ds":           pd.to_datetime(nat["date"]),
                "lower_window": -1,
                "upper_window":  1,
            })

    # Build args list
    series_keys = list(df.groupby(["store_nbr", "family"]).groups.keys())
    if max_series:
        series_keys = series_keys[:max_series]

    args_list = []
    for store_nbr, family in series_keys:
        mask      = (df["store_nbr"] == store_nbr) & (df["family"] == family)
        series_df = df[mask].sort_values("date")
        args_list.append((store_nbr, family, series_df, prophet_holidays, run_params))

    # ── Suppress cmdstanpy verbose chain logs ─────────────────
    import logging as _logging
    _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)
    _logging.getLogger("prophet").setLevel(_logging.WARNING)

    logger.info("Training Prophet  |  n_series=%d", len(args_list))
    logger.info("(cmdstanpy chain logs suppressed — watch progress every 100 series)")
    start = time.time()

    models     = {}
    val_rmsles = []
    for i, args in enumerate(args_list):
        key, model, val_r = _train_single_prophet(args)
        if model is not None:
            models[key] = model
            val_rmsles.append(val_r)
        if (i + 1) % 50 == 0:
            logger.info("Prophet: %d/%d done", i + 1, len(args_list))

    mean_r = float(np.mean(val_rmsles)) if val_rmsles else float("nan")
    logger.info(
        "Prophet done  |  n_models=%d  mean_RMSLE=%.5f  time=%.1fs",
        len(models), mean_r, time.time() - start
    )
    return models


def predict_prophet(
    models:       Dict[str, object],
    future_dates: pd.DatetimeIndex,
    oil_values:   np.ndarray = None,
) -> pd.DataFrame:
    """
    Generates Prophet predictions for all trained (store, family) series.
    Returns DataFrame: store_nbr, family, date, prophet_pred.
    """
    records = []
    for key, model in models.items():
        try:
            parts     = key.split("_", 1)
            store_nbr = int(parts[0])
            family    = parts[1]

            future = pd.DataFrame({"ds": future_dates})
            if "dcoilwtico" in model.extra_regressors:
                future["dcoilwtico"] = (
                    oil_values if oil_values is not None
                    else np.zeros(len(future_dates))
                )

            forecast = model.predict(future)
            y_pred   = np.expm1(forecast["yhat"].values).clip(0)

            for date, pred in zip(future_dates, y_pred):
                records.append({
                    "store_nbr":    store_nbr,
                    "family":       family,
                    "date":         date,
                    "prophet_pred": float(pred),
                })
        except Exception as e:
            logger.warning("Prophet predict failed for %s: %s", key, e)

    result = pd.DataFrame(records)
    logger.info("Prophet predictions  |  rows=%d", len(result))
    return result


# ============================================================
# 3. LSTM
# ============================================================

def train_lstm(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    n_features: int,
    seq_len:    int = LSTM_SEQ_LEN,
) -> Tuple:
    """
    Trains 2-layer LSTM on 3D input (samples, seq_len, n_features).

    Architecture (EDA driven):
        LSTM(128) → captures short-term patterns (day-of-week, promo)
        LSTM(64)  → captures medium-term trends (monthly seasonality)
        Dense(1)  → log1p(sales) output

    Dropout(0.2) on both LSTM layers: prevents overfitting on
    28-day sequences (small window relative to time span).

    Returns: (keras_model, val_rmsle, history)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models    import Sequential
        from tensorflow.keras.layers    import LSTM, Dense, Dropout, Input
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        logger.warning("TensorFlow not installed — skipping LSTM")
        return None, float("inf"), None

    tf.random.set_seed(RANDOM_STATE)

    logger.info(
        "Training LSTM  |  X_train=%s  seq_len=%d  n_features=%d",
        X_train.shape, seq_len, n_features
    )

    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

    callbacks = [
        EarlyStopping(
            monitor              = "val_loss",
            patience             = LSTM_PATIENCE,
            restore_best_weights = True,
            verbose              = 1,
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-6,
            verbose  = 0,
        ),
    ]

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = LSTM_EPOCHS,
        batch_size      = LSTM_BATCH_SIZE,
        callbacks       = callbacks,
        verbose         = 1,
    )

    y_log_pred = model.predict(X_val, verbose=0).flatten()
    y_pred     = np.expm1(y_log_pred).clip(0)
    y_true     = np.expm1(y_val).clip(0)
    val_rmsle  = rmsle(y_true, y_pred)

    logger.info(
        "LSTM done  |  epochs=%d  val_RMSLE=%.5f  time=%.1fs",
        len(history.history["loss"]), val_rmsle, time.time() - start
    )
    return model, val_rmsle, history


# ============================================================
# 4. ENSEMBLE — weighted average + weight tuning
# ============================================================

def tune_ensemble_weights(
    lgbm_preds:    np.ndarray,
    prophet_preds: Optional[np.ndarray],
    lstm_preds:    Optional[np.ndarray],
    y_true:        np.ndarray,
    step:          float = 0.05,
) -> dict:
    """
    Grid search over (w_lgbm, w_prophet, w_lstm) to minimize val RMSLE.
    Constraint: weights sum to 1.0.
    If prophet/lstm not available, assigns weight=0 automatically.

    Returns dict: {"lgbm": float, "prophet": float, "lstm": float, "rmsle": float}
    """
    has_prophet = prophet_preds is not None and len(prophet_preds) == len(y_true)
    has_lstm    = lstm_preds    is not None and len(lstm_preds)    == len(y_true)

    if not has_prophet and not has_lstm:
        score = rmsle(y_true, lgbm_preds)
        logger.info("Ensemble: LGBM only  |  RMSLE=%.5f", score)
        return {"lgbm": 1.0, "prophet": 0.0, "lstm": 0.0, "rmsle": round(score, 6)}

    best_rmsle   = float("inf")
    best_weights = {"lgbm": 1.0, "prophet": 0.0, "lstm": 0.0}
    steps        = np.round(np.arange(0, 1.0 + step, step), 6)

    for w_lgbm in steps:
        for w_prophet in steps:
            w_lstm = round(1.0 - w_lgbm - w_prophet, 6)
            if w_lstm < -1e-6:
                continue
            if not has_prophet and w_prophet > 1e-6:
                continue
            if not has_lstm and w_lstm > 1e-6:
                continue
            w_lstm = max(w_lstm, 0.0)

            blend = w_lgbm * lgbm_preds
            if has_prophet:
                blend = blend + w_prophet * prophet_preds
            if has_lstm:
                blend = blend + w_lstm * lstm_preds

            score = rmsle(y_true, blend.clip(0))
            if score < best_rmsle:
                best_rmsle   = score
                best_weights = {
                    "lgbm":    round(float(w_lgbm),    4),
                    "prophet": round(float(w_prophet),  4),
                    "lstm":    round(float(w_lstm),     4),
                }

    best_weights["rmsle"] = round(best_rmsle, 6)
    logger.info("Ensemble tuned  |  %s", best_weights)
    return best_weights


def ensemble_predict(
    lgbm_preds:    np.ndarray,
    prophet_preds: Optional[np.ndarray],
    lstm_preds:    Optional[np.ndarray],
    weights:       dict,
) -> np.ndarray:
    """Weighted blend of model predictions (raw sales space)."""
    blend = weights.get("lgbm", 1.0) * np.asarray(lgbm_preds, dtype=float)
    if prophet_preds is not None and weights.get("prophet", 0) > 0:
        blend += weights["prophet"] * np.asarray(prophet_preds, dtype=float)
    if lstm_preds is not None and weights.get("lstm", 0) > 0:
        blend += weights["lstm"] * np.asarray(lstm_preds, dtype=float)
    return blend.clip(0)


# ============================================================
# SAVE / LOAD ENSEMBLE CONFIG
# ============================================================

def save_ensemble_config(weights: dict, model_dir: str = MODEL_DIR) -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "ensemble_config.json")
    with open(path, "w") as f:
        json.dump(weights, f, indent=2)
    logger.info("Ensemble config saved → %s", path)
    return path


def load_ensemble_config(model_dir: str = MODEL_DIR) -> dict:
    path = os.path.join(model_dir, "ensemble_config.json")
    if not os.path.exists(path):
        logger.warning("Ensemble config not found — using defaults from config.py")
        return dict(ENSEMBLE_WEIGHTS)
    with open(path) as f:
        return json.load(f)