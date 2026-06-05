# ============================================================
# METRICS — Store Sales Forecasting System
# ============================================================
# Evaluation metrics for time series regression:
#
#   PRIMARY   : RMSLE  — Kaggle competition metric
#   SECONDARY : RMSE   — on log-space predictions
#               MAE    — on raw sales predictions
#               MAPE   — % error (skip zero-sales rows)
#               R2     — explained variance
#
# PSI for drift monitoring (same fixed version as Credit Risk):
#   PSI < 0.10   → stable
#   PSI 0.10-0.20 → moderate shift (monitor)
#   PSI > 0.20   → major shift (retrain)
#
# Champion vs Challenger comparison (adapted for regression).
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ============================================================
# RMSLE
# ============================================================

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    RMSLE = sqrt( mean( (log1p(y_pred) - log1p(y_true))^2 ) )

    Why RMSLE for retail sales:
        - Scale-invariant: small and large stores penalized equally
        - Handles zero sales via log1p (log1p(0) = 0)
        - Penalizes under-prediction more than over (retail prefers this)

    Note: if training target = log1p(sales) and we predict in log
    space, then RMSE(y_log_pred, y_log_true) == RMSLE(expm1(pred), true).
    """
    y_true = np.asarray(y_true, dtype=float).clip(0)
    y_pred = np.asarray(y_pred, dtype=float).clip(0)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def rmsle_from_log(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> float:
    """
    Shortcut: RMSLE when both inputs are already log1p-transformed.
    Used during LightGBM training (target = log1p(sales)).
    """
    return float(np.sqrt(np.mean((np.asarray(y_log_pred) - np.asarray(y_log_true)) ** 2)))


# ============================================================
# RMSE / MAE / MAPE / R2
# ============================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """
    MAPE — excludes rows where true sales < eps (zero-inflated rows).
    eps=1.0: ignore rows with < 1 unit sold (no meaningful % error).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = y_true >= eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)



# ============================================================
# WAPE — Weighted Absolute Percentage Error (Retail Industry Standard)
# ============================================================

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    WAPE = sum(|actual - predicted|) / sum(actual) * 100

    Why WAPE over MAPE for retail forecasting:
        - MAPE weights all rows equally: small stores dominate the result
        - WAPE weights by actual volume: large stores matter proportionally
        - Used by Walmart, Amazon, Tesco, Zara for demand forecasting KPIs
        - Handles zero sales naturally (zeros in sum, not division)
        - Business interpretation: "We mis-forecast X% of total sales volume"
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom  = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


# ============================================================
# BIAS — Systematic Over/Under Forecasting Detection
# ============================================================

def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Bias = mean(predicted - actual)
    Positive = over-forecast (excess inventory), Negative = under-forecast (stockouts).
    RMSLE/RMSE don't capture direction — Bias does.
    Industry KPI: |Bias%| < 5% of mean sales is acceptable.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def forecast_bias_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Bias as % of mean actual sales — scale-independent."""
    y_true      = np.asarray(y_true, dtype=float)
    y_pred      = np.asarray(y_pred, dtype=float)
    mean_actual = np.mean(y_true)
    if mean_actual == 0:
        return float("nan")
    return float(np.mean(y_pred - y_true) / mean_actual * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


# ============================================================
# EVALUATE ALL METRICS
# ============================================================

def evaluate_forecast(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    y_log_true: np.ndarray = None,
    y_log_pred: np.ndarray = None,
    model_name: str = "model",
) -> dict:
    """
    Computes all metrics for one model's forecast.

    Args:
        y_true     : raw sales (not log-transformed)
        y_pred     : predicted raw sales (expm1 of log predictions)
        y_log_true : log1p(y_true) — for log-space RMSE (optional)
        y_log_pred : log1p(y_pred) — for log-space RMSE (optional)
        model_name : label for logging

    Returns dict of all metrics.
    """
    y_true = np.asarray(y_true, dtype=float).clip(0)
    y_pred = np.asarray(y_pred, dtype=float).clip(0)

    metrics = {
        "model":      model_name,
        "rmsle":      rmsle(y_true, y_pred),
        "rmse":       rmse(y_true, y_pred),
        "mae":        mae(y_true, y_pred),
        "mape":       mape(y_true, y_pred),
        "wape":       wape(y_true, y_pred),
        "bias":       forecast_bias(y_true, y_pred),
        "bias_pct":   forecast_bias_pct(y_true, y_pred),
        "r2":         r2_score(y_true, y_pred),
        "n_samples":  int(len(y_true)),
    }

    if y_log_true is not None and y_log_pred is not None:
        metrics["log_rmse"] = rmsle_from_log(
            np.asarray(y_log_true), np.asarray(y_log_pred)
        )

    logger.info(
        "%s  |  RMSLE=%.5f  RMSE=%.2f  MAE=%.2f  MAPE=%.2f%%  "
        "WAPE=%.2f%%  Bias=%.2f (%.2f%%)  R2=%.4f",
        model_name,
        metrics["rmsle"], metrics["rmse"], metrics["mae"],
        metrics.get("mape", float("nan")),
        metrics.get("wape", float("nan")),
        metrics.get("bias", float("nan")),
        metrics.get("bias_pct", float("nan")),
        metrics["r2"],
    )
    return metrics


# ============================================================
# WALK-FORWARD CV SCORE
# ============================================================

def walk_forward_cv_score(
    model,
    df:           pd.DataFrame,
    feature_cols: list,
    preprocessor,
    splits:       list,
    model_type:   str = "lgbm",
) -> dict:
    """
    Computes average RMSLE across walk-forward validation splits.
    Only supports lgbm directly here — prophet/lstm handled in evaluation.py.

    Returns dict with mean_rmsle, std_rmsle, split_scores.
    """
    split_scores = []

    for i, (train_mask, val_mask, val_start, val_end) in enumerate(splits):
        df_val = df[val_mask].copy()

        if model_type == "lgbm":
            X_val  = preprocessor.transform(df_val[feature_cols])
            y_log  = model.predict(X_val)
            y_pred = np.expm1(y_log).clip(0)
            y_true = df_val["sales"].values
            score  = rmsle(y_true, y_pred)
            split_scores.append(score)
            logger.info(
                "CV Split %d/%d  |  val=[%s → %s]  RMSLE=%.5f",
                i + 1, len(splits), val_start, val_end, score
            )

    if not split_scores:
        return {"mean_rmsle": float("nan"), "std_rmsle": float("nan"), "split_scores": []}

    result = {
        "mean_rmsle":   float(np.mean(split_scores)),
        "std_rmsle":    float(np.std(split_scores)),
        "split_scores": split_scores,
    }

    logger.info(
        "Walk-forward CV  |  mean_RMSLE=%.5f  std=%.5f  n_splits=%d",
        result["mean_rmsle"], result["std_rmsle"], len(split_scores)
    )
    return result


# ============================================================
# PSI — Population Stability Index (same fixed version as Credit Risk)
# ============================================================

def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index for drift monitoring.

    Fixed implementation:
        1. Bin edges from reference (expected) distribution only
        2. Both distributions binned with SAME edges
        3. Compare proportions via PSI formula

    Common bug avoided: ranking both independently → PSI always ~0.
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual   = np.asarray(actual,   dtype=float)

        quantiles = np.linspace(0, 100, buckets + 1)
        bin_edges = np.percentile(expected, quantiles)
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0]  = min(bin_edges[0],  actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        exp_hist, _ = np.histogram(expected, bins=bin_edges)
        act_hist, _ = np.histogram(actual,   bins=bin_edges)

        exp_pct = exp_hist / (exp_hist.sum() + 1e-9)
        act_pct = act_hist / (act_hist.sum() + 1e-9)
        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)

        return float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))

    except Exception as e:
        logger.warning("PSI computation failed: %s", e)
        return float("nan")


# ============================================================
# DRIFT REPORT
# ============================================================

def compute_drift_report(
    df_ref:       pd.DataFrame,
    df_new:       pd.DataFrame,
    feature_cols: list,
    top_n:        int = 15,
) -> pd.DataFrame:
    """
    PSI drift report for all feature columns.
    Saved to forecast_models/feature_drift_report.csv by training_pipeline.
    """
    scores = {}
    for col in feature_cols:
        if col in df_ref.columns and col in df_new.columns:
            ref = df_ref[col].dropna().values
            new = df_new[col].dropna().values
            if len(ref) > 10 and len(new) > 10:
                scores[col] = psi(ref, new)

    drift_df = (
        pd.Series(scores)
          .sort_values(ascending=False)
          .head(top_n)
          .reset_index()
    )
    drift_df.columns = ["feature", "drift_score"]

    def _flag(v):
        if v >= 0.20: return "CRITICAL"
        if v >= 0.10: return "MODERATE"
        return "OK"

    drift_df["status"] = drift_df["drift_score"].apply(_flag)

    logger.info(
        "Drift report  |  features=%d  max_psi=%.4f",
        len(drift_df),
        drift_df["drift_score"].max() if len(drift_df) else 0
    )
    return drift_df


# ============================================================
# CHAMPION vs CHALLENGER (regression version)
# ============================================================

def compare_models(
    champion_metrics:   dict,
    challenger_metrics: dict,
    rmsle_threshold:    float = 0.005,
) -> dict:
    """
    Promotion gates for regression forecasting models:
        Gate 1: Challenger RMSLE lower than champion by >= rmsle_threshold
        Gate 2: Challenger R2 >= 0.80
        Gate 3: Challenger MAPE <= champion MAPE

    Returns result dict with decision: PROMOTED or REJECTED.
    """
    champ_rmsle = champion_metrics.get("rmsle", float("inf"))
    chal_rmsle  = challenger_metrics.get("rmsle", float("inf"))
    chal_r2     = challenger_metrics.get("r2",    0.0)
    champ_wape  = champion_metrics.get("wape",   float("inf"))
    chal_wape   = challenger_metrics.get("wape",  float("inf"))
    chal_bias   = abs(challenger_metrics.get("bias_pct", float("inf")))

    gate1 = (champ_rmsle - chal_rmsle) >= rmsle_threshold
    gate2 = chal_r2 >= 0.80
    gate3 = chal_wape <= champ_wape          # WAPE replaces MAPE (volume-weighted)

    failed = []
    if not gate1:
        diff = champ_rmsle - chal_rmsle
        failed.append(f"RMSLE improvement {diff:.5f} < {rmsle_threshold}")
    if not gate2:
        failed.append(f"R2 {chal_r2:.4f} < 0.80")
    if not gate3:
        failed.append(f"WAPE {chal_wape:.2f}% >= champion {champ_wape:.2f}%")

    return {
        "decision":          "PROMOTED" if (gate1 and gate2 and gate3) else "REJECTED",
        "reason":            "All gates passed" if (gate1 and gate2 and gate3)
                             else "Gates failed: " + " | ".join(failed),
        "champion_rmsle":    round(champ_rmsle, 6),
        "challenger_rmsle":  round(chal_rmsle,  6),
        "rmsle_improvement": round(champ_rmsle - chal_rmsle, 6),
        "challenger_r2":     round(chal_r2,  4),
        "challenger_wape":   round(chal_wape,  4),
        "challenger_bias_pct": round(chal_bias,  4),
        "gates": {
            "rmsle_passed": gate1,
            "r2_passed":    gate2,
            "wape_passed":  gate3,
        },
    }