# ============================================================
# EVALUATION — Store Sales Forecasting System
# ============================================================
# Handles:
#   - Full model evaluation (LGBM + Prophet + LSTM + Ensemble)
#   - SHAP explainability for LightGBM
#   - Model saving (joblib for LGBM, pickle for Prophet, h5 for LSTM)
#   - Model card building + saving
#   - MLflow experiment tracking
#   - Monitor scores saving (for dashboard)
# ============================================================

import os
import json
import time
import joblib
import logging
import numpy as np
import pandas as pd

from typing import Optional, Dict

from src.config   import MODEL_DIR, RANDOM_STATE
from src.metrics  import evaluate_forecast, compute_drift_report

logger = logging.getLogger(__name__)


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_all_models(
    y_true:         np.ndarray,
    lgbm_preds:     np.ndarray,
    prophet_preds:  Optional[np.ndarray],
    lstm_preds:     Optional[np.ndarray],
    ensemble_preds: np.ndarray,
    y_log_true:     np.ndarray = None,
    y_log_lgbm:     np.ndarray = None,
) -> pd.DataFrame:
    """
    Evaluates all available models and returns comparison DataFrame.

    Args:
        y_true         : raw sales ground truth
        lgbm_preds     : LightGBM raw predictions (expm1 applied)
        prophet_preds  : Prophet raw predictions (None if not trained)
        lstm_preds     : LSTM raw predictions (None if not trained)
        ensemble_preds : Weighted blend predictions
        y_log_true     : log1p(y_true) — for log-space RMSE
        y_log_lgbm     : log1p(lgbm_preds) — for log-space RMSE

    Returns: DataFrame with one row per model, all metrics as columns.
    """
    results = []

    results.append(evaluate_forecast(
        y_true, lgbm_preds,
        y_log_true=y_log_true, y_log_pred=y_log_lgbm,
        model_name="LightGBM"
    ))

    if prophet_preds is not None:
        results.append(evaluate_forecast(
            y_true, prophet_preds, model_name="Prophet"
        ))

    if lstm_preds is not None:
        results.append(evaluate_forecast(
            y_true, lstm_preds, model_name="LSTM"
        ))

    results.append(evaluate_forecast(
        y_true, ensemble_preds, model_name="Ensemble"
    ))

    df_results = pd.DataFrame(results).sort_values("rmsle").reset_index(drop=True)

    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    show_cols = [c for c in ["model","rmsle","rmse","mae","mape","wape","bias_pct","r2"] if c in df_results.columns]
    print(df_results[show_cols].to_string(index=False))
    print("=" * 70)

    return df_results


# ============================================================
# SHAP EXPLAINABILITY (LightGBM)
# ============================================================

def compute_shap(
    lgbm_model,
    X_sample:      np.ndarray,
    feature_names: list = None,
    max_samples:   int  = 500,
) -> Optional[dict]:
    """
    Computes SHAP values for LightGBM using TreeExplainer.

    Why SHAP for forecasting:
        - Explains which features drive each store-family forecast
        - Useful for debugging: why did store 5 GROCERY I spike on day X?
        - Feature importance by gain can be misleading for correlated features;
          SHAP gives a fairer attribution

    Returns dict with shap_top (top feature mean |SHAP|) or None on failure.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP")
        return None

    try:
        # Sample for speed
        idx      = np.random.RandomState(RANDOM_STATE).choice(
            len(X_sample), min(max_samples, len(X_sample)), replace=False
        )
        X_subset = X_sample[idx]

        explainer   = shap.TreeExplainer(lgbm_model)
        shap_values = explainer.shap_values(X_subset)

        mean_shap = np.abs(shap_values).mean(axis=0)

        names = feature_names if feature_names else [f"f{i}" for i in range(len(mean_shap))]
        shap_df = (
            pd.DataFrame({"feature": names, "mean_shap": mean_shap})
            .sort_values("mean_shap", ascending=False)
            .head(20)
        )

        print("\nTOP SHAP FEATURES:")
        print(shap_df.head(10).to_string(index=False))

        shap_top = dict(zip(shap_df["feature"], shap_df["mean_shap"]))
        return {"shap_top": shap_top, "shap_df": shap_df}

    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return None


# ============================================================
# SAVE MODELS
# ============================================================

def save_lgbm_model(
    model,
    model_name: str = "LightGBM",
    version:    str = "v1",
    model_dir:  str = MODEL_DIR,
) -> str:
    """Saves LightGBM model via joblib. Returns saved path."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"forecast_model_{model_name}_{version}.joblib")
    joblib.dump(model, path)
    logger.info("LightGBM saved → %s  (%.1f KB)", path, os.path.getsize(path) / 1024)
    return path


def save_prophet_models(
    models:    Dict[str, object],
    model_dir: str = MODEL_DIR,
) -> str:
    """Saves all Prophet models as a single joblib file."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "prophet_models.joblib")
    joblib.dump(models, path)
    logger.info(
        "Prophet models saved → %s  (%d series, %.1f MB)",
        path, len(models), os.path.getsize(path) / 1e6
    )
    return path


def save_lstm_model(
    model,
    model_dir: str = MODEL_DIR,
) -> str:
    """Saves Keras LSTM model in native Keras format."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "lstm_model.keras")
    try:
        model.save(path)
        logger.info("LSTM saved → %s", path)
    except Exception as e:
        logger.warning("LSTM save failed: %s", e)
        path = ""
    return path


def save_preprocessor(
    preprocessor,
    name:      str = "tree_preprocessor",
    model_dir: str = MODEL_DIR,
) -> str:
    """Saves fitted ColumnTransformer preprocessor."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{name}.joblib")
    joblib.dump(preprocessor, path)
    logger.info("Preprocessor saved → %s", path)
    return path


def save_encoding_stats(
    encoding_stats: dict,
    model_dir: str = MODEL_DIR,
) -> str:
    """Saves target encoding stats (family_mean, store_mean etc.) as JSON."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "encoding_stats.json")

    # Convert numpy types to Python native for JSON serialization
    def _to_native(obj):
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(path, "w") as f:
        json.dump(_to_native(encoding_stats), f, indent=2)
    logger.info("Encoding stats saved → %s", path)
    return path


# ============================================================
# MODEL REGISTRY
# ============================================================

def update_model_registry(
    lgbm_path:         str,
    preprocessor_path: str,
    ensemble_weights:  dict,
    metrics:           dict,
    model_dir:         str = MODEL_DIR,
    model_card_path:   str = "",
) -> str:
    """
    Updates latest_model.json — equivalent to Credit Risk project registry.
    Stores all paths needed by forecast_api.py to load the model.
    """
    # Normalize paths to forward slash — Windows os.path.join uses backslash
    # which breaks loading on Linux (Render/Docker)
    registry = {
        "lgbm_model_path":      lgbm_path.replace("\\", "/"),
        "preprocessor_path":    preprocessor_path.replace("\\", "/"),
        "ensemble_weights":     ensemble_weights,
        "rmsle":                round(metrics.get("rmsle", 0), 6),
        "r2":                   round(metrics.get("r2",    0), 4),
        "trained_at":           time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_card_path":      model_card_path.replace("\\", "/"),
    }
    path = os.path.join(model_dir, "latest_model.json")
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Model registry updated → %s", path)
    return path


def load_model_registry(model_dir: str = MODEL_DIR) -> dict:
    """Loads latest_model.json for inference."""
    path = os.path.join(model_dir, "latest_model.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model registry not found: {path}. Run train_model.py first."
        )
    with open(path) as f:
        return json.load(f)


# ============================================================
# MODEL CARD
# ============================================================

def build_model_card(
    eval_df:           pd.DataFrame,
    ensemble_weights:  dict,
    fi_df:             Optional[pd.DataFrame],
    shap_result:       Optional[dict],
    train_rows:        int,
    val_rows:          int,
    feature_cols:      list,
    drift_df:          Optional[pd.DataFrame] = None,
) -> dict:
    """
    Builds structured model card for Store Sales system.
    Saved as JSON — similar to Credit Risk model_card_LightGBM_v1.json.
    """
    # Best model = Ensemble (or LightGBM if no ensemble)
    best_row = eval_df[eval_df["model"] == "Ensemble"]
    if len(best_row) == 0:
        best_row = eval_df.iloc[0:1]

    best_metrics = best_row.iloc[0].to_dict()

    card = {
        "project":         "Store Sales Forecasting System",
        "trained_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "source":      "Kaggle — store-sales-time-series-forecasting",
            "train_rows":  train_rows,
            "val_rows":    val_rows,
            "n_features":  len(feature_cols),
        },
        "models_evaluated": eval_df[
            ["model", "rmsle", "rmse", "mae", "mape", "r2"]
        ].round(5).to_dict(orient="records"),
        "ensemble_weights": ensemble_weights,
        "best_model": {
            "name":   best_metrics.get("model", "Ensemble"),
            "rmsle":  round(float(best_metrics.get("rmsle", 0)), 6),
            "rmse":   round(float(best_metrics.get("rmse",  0)), 4),
            "mae":    round(float(best_metrics.get("mae",   0)), 4),
            "mape":   round(float(best_metrics.get("mape",  0)), 4),
            "r2":     round(float(best_metrics.get("r2",    0)), 4),
        },
        "feature_cols": feature_cols,
    }

    if fi_df is not None:
        card["feature_importances"] = dict(
            zip(fi_df["feature"].head(20), fi_df["importance"].head(20))
        )

    if shap_result is not None:
        card["shap_top_features"] = {
            k: round(float(v), 6)
            for k, v in shap_result.get("shap_top", {}).items()
        }

    if drift_df is not None:
        card["drift_report"] = drift_df.head(10).to_dict(orient="records")

    return card


def save_model_card(card: dict, model_dir: str = MODEL_DIR) -> str:
    """Saves model card JSON."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "model_card_ensemble_v1.json")
    with open(path, "w") as f:
        json.dump(card, f, indent=2, default=str)
    logger.info("Model card saved → %s", path)
    return path


# ============================================================
# MONITOR SCORES (for dashboard)
# ============================================================

def save_monitor_scores(
    df_val:         pd.DataFrame,
    ensemble_preds: np.ndarray,
    model_dir:      str = MODEL_DIR,
) -> str:
    """
    Saves validation predictions for dashboard monitoring.
    Columns: date, store_nbr, family, actual_sales, predicted_sales,
             abs_error, pct_error, confidence_band (if available)
    """
    # ── If df_val is already df_scored (has confidence_band from forecast engine)
    if "confidence_band" in df_val.columns:
        monitor = df_val.copy()

        # Rename sales → actual_sales if needed
        if "sales" in monitor.columns and "actual_sales" not in monitor.columns:
            monitor = monitor.rename(columns={"sales": "actual_sales"})

        # Ensure predicted_sales column exists
        if "predicted_sales" not in monitor.columns:
            monitor["predicted_sales"] = np.round(ensemble_preds, 2)

        # Recalculate error columns (always fresh)
        monitor["abs_error"] = np.abs(
            monitor["actual_sales"] - monitor["predicted_sales"]
        ).round(2)
        monitor["pct_error"] = np.where(
            monitor["actual_sales"] > 0,
            (monitor["abs_error"] / monitor["actual_sales"] * 100).round(2),
            0.0
        )

    else:
        # ── Fallback: raw df_val without confidence_band ──────
        monitor = df_val[["date", "store_nbr", "family", "sales"]].copy()
        monitor = monitor.rename(columns={"sales": "actual_sales"})
        monitor["predicted_sales"] = np.round(ensemble_preds, 2)
        monitor["abs_error"]       = np.abs(
            monitor["actual_sales"] - monitor["predicted_sales"]
        ).round(2)
        monitor["pct_error"]       = np.where(
            monitor["actual_sales"] > 0,
            (monitor["abs_error"] / monitor["actual_sales"] * 100).round(2),
            0.0
        )
    
    # Sirf relevant columns rakhlo
    keep_cols = ["date", "store_nbr", "family", "actual_sales", 
                 "predicted_sales", "confidence_band", "rule_triggered",
                 "abs_error", "pct_error"]
    monitor = monitor[[c for c in keep_cols if c in monitor.columns]]
    
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "monitor_scores.csv")
    monitor.to_csv(path, index=False)
    logger.info(
        "Monitor scores saved → %s  (%d rows, cols=%s)",
        path, len(monitor), monitor.columns.tolist()
    )
    return path

# ============================================================
# MLFLOW LOGGING
# ============================================================

def mlflow_log_run(
    run_name:         str,
    eval_df:          pd.DataFrame,
    ensemble_weights: dict,
    lgbm_model,
    model_card:       dict,
    X_train_sample:   np.ndarray = None,
    feature_names:    list       = None,
) -> None:
    """
    Logs metrics + model to MLflow.
    Skips gracefully if MLflow not installed or tracking fails.
    """
    try:
        import mlflow
        import mlflow.lightgbm
    except ImportError:
        logger.info("MLflow not installed — skipping MLflow logging")
        return

    try:
        with mlflow.start_run(run_name=run_name):
            # Log metrics for each model
            for _, row in eval_df.iterrows():
                prefix = row["model"].lower().replace(" ", "_")
                mlflow.log_metric(f"{prefix}_rmsle", row["rmsle"])
                mlflow.log_metric(f"{prefix}_mae",   row["mae"])
                mlflow.log_metric(f"{prefix}_r2",    row["r2"])

            # Log ensemble weights
            for model_name, weight in ensemble_weights.items():
                if isinstance(weight, (int, float)):
                    mlflow.log_param(f"weight_{model_name}", weight)

            # Log LightGBM model
            if lgbm_model is not None and X_train_sample is not None:
                try:
                    input_example = X_train_sample[:5].astype(float)
                    mlflow.lightgbm.log_model(
                        lgbm_model,
                        name           = "lgbm_model",
                        pip_requirements = ["lightgbm"],
                        input_example  = input_example,
                    )
                except Exception as e:
                    logger.warning("MLflow model log failed: %s", e)

            logger.info("MLflow run logged: %s", run_name)

    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)