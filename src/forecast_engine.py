# ============================================================
# FORECAST ENGINE — Store Sales Forecasting System
# ============================================================
# Equivalent to risk_engine.py in Credit Risk project.
# Converts raw model predictions into structured forecast output
# with business context:
#
#   CONFIDENCE BANDS:
#       HIGH    → low error on similar historical patterns
#       MEDIUM  → moderate uncertainty
#       LOW     → high uncertainty (new store, sparse family, holiday)
#
#   FORECAST FLAGS:
#       is_holiday_forecast   → prediction on/near national holiday
#       is_promo_forecast     → onpromotion > 0 on forecast date
#       is_zero_risk          → family has >50% zero sales historically
#       is_new_pattern        → date beyond training distribution
#
#   BUSINESS RULES (override model when needed):
#       - If store closed (earthquake/disaster flag) → sales = 0
#       - If family_zero_pct > 0.90 AND no promo → floor prediction to 0
#       - If prediction < 0 → clip to 0 (always)
# ============================================================

import numpy as np
import pandas as pd
import logging

from src.config import EARTHQUAKE_DATE, EARTHQUAKE_WINDOW_POST

logger = logging.getLogger(__name__)


# ============================================================
# CONFIDENCE BAND ASSIGNMENT
# ============================================================

def get_confidence_band(
    pred:           float,
    family_zero_pct: float = 0.0,
    is_holiday:     int   = 0,
    is_promoted:    int   = 0,
    days_since_train_end: int = 0,
) -> str:
    """
    Assigns a confidence band to a single prediction.

    HIGH   → straightforward prediction (non-holiday, non-promo,
              low zero-rate family, not far from training window)
    MEDIUM → some uncertainty factor present
    LOW    → multiple uncertainty factors (holiday + promo + high zero-rate)
             OR very far from training data

    This is the Store Sales equivalent of risk_band (LOW/MEDIUM/HIGH)
    in Credit Risk.
    """
    uncertainty_score = 0

    # High zero-rate families are hard to predict
    if family_zero_pct > 0.50:
        uncertainty_score += 1
    if family_zero_pct > 0.80:
        uncertainty_score += 1

    # Holidays introduce demand spikes (harder to predict magnitude)
    if is_holiday:
        uncertainty_score += 1

    # Promotions increase variability
    if is_promoted:
        uncertainty_score += 1

    # Predictions far from training window are less reliable
    if days_since_train_end > 30:
        uncertainty_score += 2
    elif days_since_train_end > 7:
        uncertainty_score += 1

    if uncertainty_score == 0:
        return "HIGH"
    elif uncertainty_score <= 2:
        return "MEDIUM"
    else:
        return "LOW"


# ============================================================
# BUSINESS RULES
# ============================================================

def apply_business_rules(
    pred:            float,
    family_zero_pct: float = 0.0,
    is_promoted:     int   = 0,
    is_store_closed: int   = 0,
    rule_log:        list  = None,
) -> tuple:
    """
    Applies hard business rules to model prediction.
    Rules checked BEFORE returning final prediction — same hierarchy
    as Credit Risk rule engine (rules override ML).

    Args:
        pred            : raw model prediction (sales units)
        family_zero_pct : fraction of zero-sales days for this family
        is_promoted     : 1 if items on promotion on forecast date
        is_store_closed : 1 if store known to be closed (disaster, etc.)
        rule_log        : list to append triggered rule name to

    Returns:
        (adjusted_pred, rule_triggered_name or None)
    """
    rule_triggered = None

    # Rule 1: Store closed → sales = 0 (hard override)
    if is_store_closed:
        rule_triggered = "STORE_CLOSED"
        if rule_log is not None:
            rule_log.append(rule_triggered)
        return 0.0, rule_triggered

    # Rule 2: Near-dead family with no promotion → floor to 0
    if family_zero_pct > 0.90 and not is_promoted:
        rule_triggered = "HIGH_ZERO_FAMILY_NO_PROMO"
        if rule_log is not None:
            rule_log.append(rule_triggered)
        return 0.0, rule_triggered

    # Rule 3: Negative prediction → clip to 0 (always)
    if pred < 0:
        rule_triggered = "NEGATIVE_CLIP"
        if rule_log is not None:
            rule_log.append(rule_triggered)
        return 0.0, rule_triggered

    return float(pred), rule_triggered


# ============================================================
# SCORE SINGLE FORECAST (for API)
# ============================================================

def score_forecast(
    pred:               float,
    store_nbr:          int,
    family:             str,
    forecast_date:      pd.Timestamp,
    train_end_date:     pd.Timestamp,
    family_zero_pct:    float = 0.0,
    is_holiday:         int   = 0,
    is_promoted:        int   = 0,
    is_store_closed:    int   = 0,
) -> dict:
    """
    Full forecast scoring for a single (store, family, date) prediction.
    Used by forecast_api.py /forecast endpoint.

    Returns structured dict:
        predicted_sales     : final adjusted prediction
        confidence_band     : HIGH / MEDIUM / LOW
        rule_triggered      : business rule name or None
        is_holiday_forecast : bool
        is_promo_forecast   : bool
        days_ahead          : how many days from training end
    """
    days_since_train_end = max(0, (forecast_date - train_end_date).days)

    # Apply business rules first
    adjusted_pred, rule_triggered = apply_business_rules(
        pred            = pred,
        family_zero_pct = family_zero_pct,
        is_promoted     = is_promoted,
        is_store_closed = is_store_closed,
    )

    # Confidence band
    confidence = get_confidence_band(
        pred                 = adjusted_pred,
        family_zero_pct      = family_zero_pct,
        is_holiday           = is_holiday,
        is_promoted          = is_promoted,
        days_since_train_end = days_since_train_end,
    )

    return {
        "store_nbr":          int(store_nbr),
        "family":             str(family),
        "forecast_date":      str(forecast_date.date()),
        "predicted_sales":    round(adjusted_pred, 2),
        "confidence_band":    confidence,
        "rule_triggered":     rule_triggered,
        "is_holiday_forecast": bool(is_holiday),
        "is_promo_forecast":   bool(is_promoted),
        "days_ahead":         int(days_since_train_end),
    }


# ============================================================
# BATCH FORECAST ENGINE (for training pipeline)
# ============================================================

def run_forecast_engine(
    df_forecast:     pd.DataFrame,
    predictions:     np.ndarray,
    train_end_date:  pd.Timestamp,
    encoding_stats:  dict,
) -> pd.DataFrame:
    """
    Applies business rules + confidence bands to a full batch
    of predictions. Used by training_pipeline.py.

    Args:
        df_forecast     : DataFrame with store_nbr, family, date,
                          onpromotion, is_any_holiday, is_earthquake_post,
                          family_zero_pct columns
        predictions     : raw model predictions (same row order as df_forecast)
        train_end_date  : last date in training data
        encoding_stats  : from feature_engineering (has family_zero_pct)

    Returns:
        df_forecast with added columns:
            predicted_sales, confidence_band, rule_triggered
    """
    df = df_forecast.copy()
    df["raw_prediction"]  = predictions
    df["predicted_sales"] = 0.0
    df["confidence_band"] = "MEDIUM"
    df["rule_triggered"]  = None

    family_zero_pct = encoding_stats.get("family_zero_pct", {})

    for idx, row in df.iterrows():
        store_nbr    = row.get("store_nbr", 0)
        family       = row.get("family", "")
        forecast_date = pd.Timestamp(row["date"])
        pred          = float(row["raw_prediction"])
        is_holiday    = int(row.get("is_any_holiday", 0))
        is_promoted   = int(row.get("is_promoted", int(row.get("onpromotion", 0) > 0)))
        is_closed     = int(row.get("is_earthquake_post", 0))  # treat earthquake post as uncertainty
        zero_pct      = float(family_zero_pct.get(family, 0.0))

        days_since = max(0, (forecast_date - train_end_date).days)

        adj_pred, rule = apply_business_rules(
            pred            = pred,
            family_zero_pct = zero_pct,
            is_promoted     = is_promoted,
            is_store_closed = 0,       # only hard-close on actual closure data
        )

        band = get_confidence_band(
            pred                 = adj_pred,
            family_zero_pct      = zero_pct,
            is_holiday           = is_holiday,
            is_promoted          = is_promoted,
            days_since_train_end = days_since,
        )

        df.at[idx, "predicted_sales"] = round(adj_pred, 2)
        df.at[idx, "confidence_band"] = band
        df.at[idx, "rule_triggered"]  = rule

    # Summary
    band_counts = df["confidence_band"].value_counts()
    rule_counts = df["rule_triggered"].dropna().value_counts()

    logger.info("Forecast engine done  |  rows=%d", len(df))
    logger.info("Confidence bands:\n%s", band_counts.to_string())
    if len(rule_counts) > 0:
        logger.info("Rules triggered:\n%s", rule_counts.to_string())
        
        try:
            import os
            from src.config import MODEL_DIR
            scored_path = os.path.join(MODEL_DIR, "df_scored.csv")
            df.to_csv(scored_path, index=False)
            logger.info("df_scored saved → %s", scored_path)
        except Exception as e:
            logger.warning("df_scored save failed: %s", e)

    return df


# ============================================================
# STORE-LEVEL FORECAST SUMMARY
# ============================================================

def store_family_summary(df_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates predictions to store-family level for dashboard.

    Returns DataFrame with:
        store_nbr, family, total_predicted, mean_confidence, n_rules_triggered
    """
    summary = df_scored.groupby(["store_nbr", "family"]).agg(
        total_predicted    = ("predicted_sales", "sum"),
        mean_predicted     = ("predicted_sales", "mean"),
        n_days             = ("predicted_sales", "count"),
        n_rules_triggered  = ("rule_triggered",  lambda x: x.notna().sum()),
        dominant_confidence = ("confidence_band", lambda x: x.mode().iloc[0] if len(x) > 0 else "MEDIUM"),
    ).reset_index()

    logger.info(
        "Store-family summary  |  rows=%d  total_predicted=%.0f",
        len(summary), summary["total_predicted"].sum()
    )
    return summary
