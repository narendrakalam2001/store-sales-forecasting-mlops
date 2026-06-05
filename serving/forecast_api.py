# ============================================================
# FORECAST API — Store Sales Forecasting System
# ============================================================
# FastAPI endpoints:
#
#   GET  /           → home message
#   GET  /health     → model load status
#   GET  /model_info → registry + metrics
#   POST /forecast   → single (store, family, date) forecast
#   POST /forecast_batch → multiple rows forecast
#   GET  /stores     → list of store metadata
#   GET  /families   → list of product families
#
# Same architecture as Credit Risk API (credit_risk_api.py)
# Loads model from latest_model.json on startup.
# Logs each prediction to logs/prediction_logs.csv.
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import logging
import time
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "Store Sales Forecast API",
    description = "LightGBM + Prophet + LSTM ensemble for Corporacion Favorita sales",
    version     = "1.0.0",
)

# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "forecast_models")
LOG_DIR   = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ── Global state — loaded on startup ─────────────────────────
_lgbm_model       = None
_tree_preprocessor = None
_encoding_stats   = None
_ensemble_weights = None
_registry         = None
_train_end_date   = None


# ============================================================
# STARTUP — load model
# ============================================================

@app.on_event("startup")
def load_model():
    global _lgbm_model, _tree_preprocessor, _encoding_stats
    global _ensemble_weights, _registry, _train_end_date

    try:
        import joblib

        registry_path = os.path.join(MODEL_DIR, "latest_model.json")
        if not os.path.exists(registry_path):
            logger.error("Model registry not found: %s", registry_path)
            return

        with open(registry_path) as f:
            _registry = json.load(f)

        _lgbm_model       = joblib.load(_registry["lgbm_model_path"])
        _tree_preprocessor = joblib.load(_registry["preprocessor_path"])
        _ensemble_weights = _registry.get("ensemble_weights", {"lgbm": 1.0})

        enc_path = os.path.join(MODEL_DIR, "encoding_stats.json")
        if os.path.exists(enc_path):
            with open(enc_path) as f:
                _encoding_stats = json.load(f)

        # Train end date from model card
        card_path = _registry.get("model_card_path", "")
        if os.path.exists(card_path):
            with open(card_path) as f:
                card = json.load(f)
            _train_end_date = pd.Timestamp(card.get("trained_at", "2017-08-15")[:10])
        else:
            _train_end_date = pd.Timestamp("2017-08-15")

        logger.info(
            "Model loaded  |  RMSLE=%.5f  weights=%s",
            _registry.get("rmsle", 0), _ensemble_weights
        )

    except Exception as e:
        logger.error("Model loading failed: %s", e)


# ============================================================
# INPUT SCHEMA
# ============================================================

class ForecastRequest(BaseModel):
    store_nbr:    int   = Field(..., ge=1, le=54, description="Store number (1-54)")
    family:       str   = Field(..., description="Product family (e.g. GROCERY I)")
    date:         str   = Field(..., description="Forecast date (YYYY-MM-DD)")
    onpromotion:  int   = Field(0, ge=0, description="Items on promotion")

    # Optional context features
    dcoilwtico:   Optional[float] = Field(None, description="Oil price (WTI)")
    is_national_holiday: Optional[int] = Field(0, description="1 if national holiday")
    transactions: Optional[float] = Field(None, description="Store daily transactions")

    class Config:
        json_schema_extra = {
            "example": {
                "store_nbr":   1,
                "family":      "GROCERY I",
                "date":        "2017-08-16",
                "onpromotion": 5,
                "dcoilwtico":  47.5,
                "is_national_holiday": 0,
            }
        }


class BatchForecastRequest(BaseModel):
    rows: List[ForecastRequest]


# ============================================================
# FEATURE BUILDER FOR API INFERENCE
# ============================================================

def _build_inference_features(req: ForecastRequest) -> pd.DataFrame:
    """
    Builds a single-row feature DataFrame for inference.
    Fills missing lag/rolling features with encoding stat means
    (best we can do without historical data at inference time).
    """
    date = pd.Timestamp(req.date)

    enc = _encoding_stats or {}

    family_mean  = enc.get("family_mean",  {}).get(req.family,    enc.get("global_mean", 5.0))
    store_mean   = enc.get("store_mean",   {}).get(req.store_nbr,  enc.get("global_mean", 5.0))
    cluster_mean = enc.get("store_cluster_mean", {})
    zero_pct     = enc.get("family_zero_pct", {}).get(req.family, 0.0)
    family_enc   = enc.get("family_label_map", {}).get(req.family, -1)

    # Calendar
    dow     = date.dayofweek
    month   = date.month
    year    = date.year
    oil_val = req.dcoilwtico if req.dcoilwtico is not None else 50.0
    txn_val = req.transactions if req.transactions is not None else 2000.0

    row = {
        # Store metadata
        "store_nbr":        req.store_nbr,
        "family_enc":       family_enc,
        "family_sales_mean": family_mean,
        "store_sales_mean":  store_mean,
        "family_zero_pct":   zero_pct,

        # Calendar
        "day_of_week":    dow,
        "day_of_month":   date.day,
        "month":          month,
        "year":           year,
        "quarter":        date.quarter,
        "week_of_year":   date.isocalendar()[1],
        "is_weekend":     int(dow >= 5),
        "is_month_start": int(date.is_month_start),
        "is_month_end":   int(date.is_month_end),
        "is_payday":      int(date.day in [1, 15]),
        "days_since_start": (date - pd.Timestamp("2013-01-01")).days,

        # Promotion
        "onpromotion":    req.onpromotion,
        "is_promoted":    int(req.onpromotion > 0),
        "promo_lag_1":    req.onpromotion,
        "promo_lag_7":    req.onpromotion,
        "promo_rolling_7d": req.onpromotion,
        "promo_x_lag7":   req.onpromotion * family_mean,

        # Holiday
        "is_national_holiday": req.is_national_holiday or 0,
        "is_any_holiday":      req.is_national_holiday or 0,
        "is_regional_holiday": 0,
        "is_local_holiday":    0,
        "is_holiday_event":    0,
        "is_transferred":      0,
        "is_work_day":         0,
        "holiday_count":       req.is_national_holiday or 0,
        "days_to_holiday":     1 if req.is_national_holiday else 15,
        "days_after_holiday":  1 if req.is_national_holiday else 15,

        # Oil
        "dcoilwtico":       oil_val,
        "oil_rolling_7d":   oil_val,
        "oil_rolling_28d":  oil_val,
        "oil_pct_change_7d": 0.0,
        "oil_regime_high":  int(oil_val >= 80),

        # Transactions
        "transactions":     txn_val,
        "txn_lag_1":        txn_val,
        "txn_lag_7":        txn_val,
        "txn_rolling_7d":   txn_val,
        "txn_rolling_28d":  txn_val,

        # Lag features — use family mean as proxy
        "sales_lag_1":      family_mean,
        "sales_lag_7":      family_mean,
        "sales_lag_14":     family_mean,
        "sales_lag_28":     family_mean,
        "log1p_lag_1":      np.log1p(family_mean),
        "log1p_lag_7":      np.log1p(family_mean),
        "log1p_lag_14":     np.log1p(family_mean),
        "log1p_lag_28":     np.log1p(family_mean),

        # Rolling features
        "rolling_mean_7d":  family_mean,
        "rolling_mean_14d": family_mean,
        "rolling_mean_28d": family_mean,
        "rolling_std_7d":   0.0,
        "rolling_std_14d":  0.0,
        "rolling_std_28d":  0.0,
        "ewm_alpha_07":     family_mean,
        "ewm_alpha_03":     family_mean,

        # Earthquake
        "is_earthquake_pre":  0,
        "is_earthquake_post": 0,

        # Store type (default median — type C)
        "store_type_enc":  2,
        "cluster":         8,
        "cluster_sales_mean": list(cluster_mean.values())[0] if cluster_mean else family_mean,
    }

    return pd.DataFrame([row])


# ============================================================
# PREDICT HELPER
# ============================================================

def _predict_one(req: ForecastRequest) -> dict:
    """Runs inference for one ForecastRequest."""
    if _lgbm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feat_df = _build_inference_features(req)

    # Keep only columns the preprocessor knows
    try:
        X = _tree_preprocessor.transform(feat_df)
    except Exception as e:
        # Fallback: let LightGBM handle directly if transformer fails
        logger.warning("Preprocessor transform failed: %s — using raw features", e)
        X = feat_df.values

    y_log  = float(_lgbm_model.predict(X)[0])
    y_pred = float(np.expm1(y_log))

    # LGBM weight — blend is approximate without prophet/lstm at inference
    w_lgbm = _ensemble_weights.get("lgbm", 1.0) if _ensemble_weights else 1.0
    final_pred = max(0.0, y_pred * (1.0 / w_lgbm if w_lgbm < 1.0 else 1.0))

    # Forecast engine logic
    from src.forecast_engine import score_forecast
    enc          = _encoding_stats or {}
    zero_pct     = enc.get("family_zero_pct", {}).get(req.family, 0.0)
    days_ahead   = max(0, (pd.Timestamp(req.date) - _train_end_date).days)

    scored = score_forecast(
        pred              = final_pred,
        store_nbr         = req.store_nbr,
        family            = req.family,
        forecast_date     = pd.Timestamp(req.date),
        train_end_date    = _train_end_date,
        family_zero_pct   = float(zero_pct),
        is_holiday        = req.is_national_holiday or 0,
        is_promoted       = int(req.onpromotion > 0),
        is_store_closed   = 0,
    )

    return scored


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Store Sales Forecast API is live 🛒",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health")
def health():
    return {
        "status":       "running",
        "model_loaded": _lgbm_model is not None,
        "rmsle":        _registry.get("rmsle") if _registry else None,
    }


@app.get("/model_info")
def model_info():
    if not _registry:
        raise HTTPException(status_code=404, detail="Model registry not found")
    return _registry


@app.get("/stores")
def list_stores():
    """Returns store numbers 1-54."""
    return {"stores": list(range(1, 55)), "count": 54}


@app.get("/families")
def list_families():
    """Returns all 33 product families."""
    families = [
        "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS",
        "BREAD/BAKERY", "CELEBRATION", "CLEANING", "DAIRY", "DELI",
        "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE",
        "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME APPLIANCES",
        "HOME CARE", "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE",
        "LIQUOR,WINE,BEER", "MAGAZINES", "MEATS", "PERSONAL CARE",
        "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POLO SHIRTS",
        "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES",
        "SEAFOOD",
    ]
    return {"families": families, "count": len(families)}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    """
    Single-row forecast for one (store, family, date).

    Returns:
        predicted_sales, confidence_band, rule_triggered,
        is_holiday_forecast, is_promo_forecast, days_ahead, latency_seconds
    """
    t_start = time.time()

    result = _predict_one(req)
    result["latency_seconds"] = round(time.time() - t_start, 4)

    # Log prediction
    _log_prediction({
        "timestamp":       time.time(),
        "store_nbr":       req.store_nbr,
        "family":          req.family,
        "date":            req.date,
        "onpromotion":     req.onpromotion,
        "predicted_sales": result["predicted_sales"],
        "confidence_band": result["confidence_band"],
        "rule_triggered":  result.get("rule_triggered"),
    })

    return result


@app.post("/forecast_batch")
def forecast_batch(batch: BatchForecastRequest):
    """
    Batch forecast for multiple (store, family, date) rows.
    Returns list of predictions in same order as input.
    """
    t_start = time.time()
    results = []
    for req in batch.rows:
        try:
            r = _predict_one(req)
            results.append(r)
        except Exception as e:
            results.append({
                "store_nbr":      req.store_nbr,
                "family":         req.family,
                "forecast_date":  req.date,
                "error":          str(e),
            })

    return {
        "predictions":      results,
        "count":            len(results),
        "latency_seconds":  round(time.time() - t_start, 4),
    }


# ============================================================
# PREDICTION LOGGER
# ============================================================

def _log_prediction(record: dict) -> None:
    """Appends prediction to logs/prediction_logs.csv."""
    log_path = os.path.join(LOG_DIR, "prediction_logs.csv")
    log_df   = pd.DataFrame([record])
    try:
        if os.path.exists(log_path):
            log_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False)
    except Exception as e:
        logger.warning("Prediction log failed: %s", e)
