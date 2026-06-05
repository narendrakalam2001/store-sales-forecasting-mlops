# ============================================================
# PREDICTION SERVICE — Store Sales Forecasting System
# ============================================================

import numpy as np
import pandas as pd
import logging

from src.forecast_engine import score_forecast

logger = logging.getLogger(__name__)


def prepare_features(input_data: dict, encoding_stats: dict) -> pd.DataFrame:
    """Mirrors inference feature building from forecast_api._build_inference_features."""
    from serving.forecast_api import _build_inference_features
    from pydantic import BaseModel

    class _Req(BaseModel):
        store_nbr: int
        family:    str
        date:      str
        onpromotion: int = 0
        dcoilwtico:  float = None
        is_national_holiday: int = 0
        transactions: float = None

    req = _Req(**input_data)
    return _build_inference_features(req)


def predict_single(
    model,
    preprocessor,
    encoding_stats: dict,
    input_data:     dict,
    ensemble_weights: dict,
    train_end_date: pd.Timestamp,
) -> dict:
    """
    Full prediction flow for one (store, family, date):
        1. Build feature row
        2. Preprocess
        3. LGBM predict → expm1
        4. Forecast engine (confidence + rules)
        5. Return structured output
    """
    from serving.forecast_api import _build_inference_features
    from pydantic import BaseModel

    class _Req(BaseModel):
        store_nbr: int
        family:    str
        date:      str
        onpromotion:         int   = 0
        dcoilwtico:          float = None
        is_national_holiday: int   = 0
        transactions:        float = None

    req  = _Req(**input_data)
    feat = _build_inference_features(req)

    try:
        X = preprocessor.transform(feat)
    except Exception:
        X = feat.values

    y_log = float(model.predict(X)[0])
    y_pred = float(np.expm1(y_log))
    w_lgbm = ensemble_weights.get("lgbm", 1.0)
    final  = max(0.0, y_pred * (1.0 / w_lgbm if w_lgbm < 1.0 else 1.0))

    zero_pct = (encoding_stats or {}).get("family_zero_pct", {}).get(req.family, 0.0)

    result = score_forecast(
        pred            = final,
        store_nbr       = req.store_nbr,
        family          = req.family,
        forecast_date   = pd.Timestamp(req.date),
        train_end_date  = train_end_date,
        family_zero_pct = float(zero_pct),
        is_holiday      = req.is_national_holiday or 0,
        is_promoted     = int(req.onpromotion > 0),
        is_store_closed = 0,
    )

    logger.info(
        "Prediction  |  store=%d  family=%s  date=%s  "
        "pred=%.2f  band=%s",
        req.store_nbr, req.family, req.date,
        result["predicted_sales"], result["confidence_band"]
    )
    return result
