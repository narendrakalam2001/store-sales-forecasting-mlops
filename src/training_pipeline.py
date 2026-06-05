# ============================================================
# TRAINING PIPELINE — Store Sales Forecasting System
# ============================================================
# Full end-to-end orchestration — 20 steps:
#
#  Step 1  : Load all 6 raw files
#  Step 2  : Validate + merge (stores, oil, holidays, transactions)
#  Step 3  : Feature engineering (lag, rolling, calendar, promo, oil, txn)
#  Step 4  : Drop NaN lag rows (first 28 days per store-family)
#  Step 5  : Walk-forward train/val split
#  Step 6  : Classify feature columns (binary/skewed/normal/ordinal)
#  Step 7  : Build tree preprocessor (LightGBM path)
#  Step 8  : Prepare LightGBM arrays
#  Step 9  : Train LightGBM
#  Step 10 : Train Prophet (optional)
#  Step 11 : Build LSTM preprocessor + sequences (optional)
#  Step 12 : Train LSTM (optional)
#  Step 13 : Generate val predictions from all models
#  Step 14 : Tune ensemble weights
#  Step 15 : Evaluate all models
#  Step 16 : Compute SHAP
#  Step 17 : PSI drift report
#  Step 18 : Run forecast engine (confidence bands + rules)
#  Step 19 : Save all models + encoding stats + preprocessors
#  Step 20 : Model card + registry update + MLflow
# ============================================================

import os
import sys
import time
import logging
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    RANDOM_STATE, MODEL_DIR, LSTM_SEQ_LEN, FORECAST_HORIZON,
)
from src.data_loader import (
    load_raw_files, build_merged_dataset, process_holidays,
)
from src.feature_engineering import build_features, get_feature_columns
from src.preprocessing import (
    classify_feature_columns,
    build_tree_preprocessor,
    build_lstm_preprocessor,
    drop_lag_nans,
    prepare_lgbm_arrays,
    prepare_lstm_sequences,
    walk_forward_splits,
)
from src.metrics import compute_drift_report
from src.leakage_check import detect_leakage
from src.model_tuning import (
    train_lightgbm,
    train_prophet_models,
    predict_prophet,
    train_lstm,
    tune_ensemble_weights,
    ensemble_predict,
    save_ensemble_config,
)
from src.evaluation import (
    evaluate_all_models,
    compute_shap,
    save_lgbm_model,
    save_prophet_models,
    save_lstm_model,
    save_preprocessor,
    save_encoding_stats,
    update_model_registry,
    build_model_card,
    save_model_card,
    save_monitor_scores,
    mlflow_log_run,
)
from src.forecast_engine import run_forecast_engine

np.random.seed(RANDOM_STATE)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Suppress noisy third-party loggers ───────────────────────
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)


# ============================================================
# PIPELINE CONFIG
# ============================================================

def get_pipeline_config() -> dict:
    return {
        "train_prophet":      True,
        "train_lstm":         False,    # set True if TensorFlow installed
        "prophet_max_series": None,     # None = all 1782 series
        "n_cv_splits":        3,
        "val_days":           16,       # matches Kaggle test horizon
        "add_cyclical":       False,    # True adds sin/cos (LSTM needs it)
        "compute_shap":       True,
        "shap_samples":       1000,
        "mlflow_run_name":    f"store_sales_{time.strftime('%Y%m%d_%H%M%S')}",
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_training(config: dict = None) -> dict:
    cfg   = {**get_pipeline_config(), **(config or {})}
    start = time.time()

    logger.info("=" * 65)
    logger.info("STORE SALES FORECASTING — TRAINING PIPELINE START")
    logger.info("=" * 65)

    # ── Step 1: Load raw files ────────────────────────────────
    logger.info("\n[STEP 1] Loading raw files ...")
    files = load_raw_files()

    # ── Step 2: Validate + merge ──────────────────────────────
    logger.info("\n[STEP 2] Merging all files ...")
    df             = build_merged_dataset(files, is_train=True)
    holidays_df    = process_holidays(files["holidays"])
    train_end_date = df["date"].max()
    logger.info(
        "Merged  |  shape=%s  date_range=%s→%s",
        df.shape, df["date"].min().date(), train_end_date.date()
    )

    # ── Step 3: Feature engineering ───────────────────────────
    logger.info("\n[STEP 3] Feature engineering ...")
    add_cyc = cfg["add_cyclical"] or cfg["train_lstm"]
    df, encoding_stats = build_features(df, is_train=True, add_cyclical=add_cyc)
    logger.info("Features done  |  shape=%s", df.shape)

    # ── Step 4: Drop NaN lag rows ─────────────────────────────
    logger.info("\n[STEP 4] Dropping NaN lag rows ...")
    df = drop_lag_nans(df, lag_col="sales_lag_28")

    # ── Step 5: Walk-forward splits ───────────────────────────
    logger.info("\n[STEP 5] Building walk-forward splits ...")
    splits = walk_forward_splits(df, n_splits=cfg["n_cv_splits"], val_days=cfg["val_days"])
    train_mask, val_mask, val_start, val_end = splits[-1]
    df_train = df[train_mask].copy()
    df_val   = df[val_mask].copy()
    logger.info(
        "Primary split  |  train=%d  val=%d  val=[%s→%s]",
        len(df_train), len(df_val), val_start, val_end
    )

    # ── Step 5b: Leakage check ───────────────────────────────
    logger.info("\n[STEP 5b] Running leakage check ...")
    leakage_warnings = detect_leakage(
        X_train = df_train.drop(columns=["target", "sales"], errors="ignore"),
        y_train = df_train["target"] if "target" in df_train.columns else df_train["sales"],
        threshold_corr = 0.95,
    )
    if leakage_warnings:
        logger.warning("Leakage check: %d warning(s) found — review before deploying", len(leakage_warnings))
    else:
        logger.info("Leakage check: PASSED ✅")

    # ── Step 6: Classify features ─────────────────────────────
    logger.info("\n[STEP 6] Classifying feature columns ...")
    feature_cols = get_feature_columns(df_train)
    col_groups   = classify_feature_columns(df_train, feature_cols)
    logger.info("Total features: %d", len(feature_cols))

    # ── Step 7: Build tree preprocessor ──────────────────────
    logger.info("\n[STEP 7] Building tree preprocessor ...")
    tree_pre = build_tree_preprocessor(col_groups)

    # ── Step 8: Prepare LGBM arrays ──────────────────────────
    logger.info("\n[STEP 8] Preparing LightGBM arrays ...")
    X_train, y_train, X_val, y_val, tree_pre = prepare_lgbm_arrays(
        df_train, df_val, feature_cols, tree_pre
    )

    # ── Step 9: Train LightGBM ────────────────────────────────
    logger.info("\n[STEP 9] Training LightGBM ...")
    lgbm_model, lgbm_rmsle, fi_df = train_lightgbm(
        X_train, y_train, X_val, y_val, feature_names=feature_cols
    )
    logger.info("LightGBM val_RMSLE=%.5f", lgbm_rmsle)

    # ── Step 10: Train Prophet (optional) ─────────────────────
    prophet_models = {}
    if cfg["train_prophet"]:
        logger.info("\n[STEP 10] Training Prophet ...")
        prophet_models = train_prophet_models(
            df_train, holidays_df,
            max_series=cfg["prophet_max_series"]
        )
    else:
        logger.info("\n[STEP 10] Prophet skipped")

    # ── Step 11: LSTM prep (optional) ─────────────────────────
    lstm_pre = None
    X_seq_train = X_seq_val = y_seq_train = y_seq_val = None

    if cfg["train_lstm"]:
        logger.info("\n[STEP 11] Building LSTM sequences ...")
        lstm_pre = build_lstm_preprocessor(col_groups)
        lstm_feats = get_feature_columns(df_train)

        X_seq_train, y_seq_train, lstm_pre = prepare_lstm_sequences(
            df_train, lstm_feats, LSTM_SEQ_LEN, lstm_pre, is_train=True
        )
        X_seq_val, _ = prepare_lstm_sequences(
            df_val, lstm_feats, LSTM_SEQ_LEN, lstm_pre, is_train=False
        )
        y_seq_val = df_val.iloc[LSTM_SEQ_LEN:]["target"].values.astype(np.float32)
    else:
        logger.info("\n[STEP 11] LSTM skipped")

    # ── Step 12: Train LSTM (optional) ───────────────────────
    lstm_model    = None
    lstm_rmsle    = float("inf")

    if cfg["train_lstm"] and X_seq_train is not None:
        logger.info("\n[STEP 12] Training LSTM ...")
        n_feats = X_seq_train.shape[2]
        lstm_model, lstm_rmsle, _ = train_lstm(
            X_seq_train, y_seq_train,
            X_seq_val,   y_seq_val,
            n_features=n_feats, seq_len=LSTM_SEQ_LEN
        )
        logger.info("LSTM val_RMSLE=%.5f", lstm_rmsle)
    else:
        logger.info("\n[STEP 12] LSTM training skipped")

    # ── Step 13: Val predictions from all models ──────────────
    logger.info("\n[STEP 13] Generating validation predictions ...")

    y_log_lgbm = lgbm_model.predict(X_val)
    lgbm_preds = np.expm1(y_log_lgbm).clip(0)

    # Prophet predictions
    prophet_preds = None
    if prophet_models:
        future_dates = pd.DatetimeIndex(df_val["date"].unique())
        oil_vals     = (
            df_val.groupby("date")["dcoilwtico"]
            .first()
            .reindex(future_dates)
            .values
        )
        prophet_df = predict_prophet(prophet_models, future_dates, oil_vals)

        if len(prophet_df) == 0:
            logger.warning("Prophet predictions empty — all series failed. Using LGBM only.")
            prophet_preds = None
        else:
            df_val_r = df_val.reset_index(drop=True)
            df_val_r["_idx"] = df_val_r.index
            merged_p = df_val_r.merge(
                prophet_df[["store_nbr", "family", "date", "prophet_pred"]],
                on=["store_nbr", "family", "date"], how="left"
            ).set_index("_idx")
            # fillna needs Series/scalar — convert lgbm_preds to Series for NaN fill
            lgbm_series   = pd.Series(lgbm_preds, index=merged_p.index)
            prophet_preds = merged_p["prophet_pred"].fillna(lgbm_series).values
            logger.info(
                "Prophet aligned  |  matched=%d  fallback_to_lgbm=%d",
                merged_p["prophet_pred"].notna().sum(),
                merged_p["prophet_pred"].isna().sum()
            )

    # LSTM predictions
    lstm_preds = None
    if lstm_model is not None and X_seq_val is not None:
        y_log_lstm = lstm_model.predict(X_seq_val, verbose=0).flatten()
        raw_lstm   = np.expm1(y_log_lstm).clip(0)

        # Pad front (LSTM sequences offset by seq_len)
        pad        = np.full(len(df_val) - len(raw_lstm), np.nan)
        lstm_preds = np.concatenate([pad, raw_lstm])
        mask_nan   = np.isnan(lstm_preds)
        lstm_preds[mask_nan] = lgbm_preds[mask_nan]

    y_true_val = df_val["sales"].values

    # ── Step 14: Tune ensemble weights ────────────────────────
    logger.info("\n[STEP 14] Tuning ensemble weights ...")
    ensemble_weights = tune_ensemble_weights(
        lgbm_preds, prophet_preds, lstm_preds, y_true_val, step=0.05
    )
    ensemble_preds = ensemble_predict(
        lgbm_preds, prophet_preds, lstm_preds, ensemble_weights
    )
    logger.info("Best weights: %s", ensemble_weights)

    # ── Step 15: Evaluate all ─────────────────────────────────
    logger.info("\n[STEP 15] Evaluating all models ...")
    eval_df = evaluate_all_models(
        y_true_val, lgbm_preds, prophet_preds,
        lstm_preds, ensemble_preds,
        y_log_true=y_val, y_log_lgbm=y_log_lgbm
    )
    best_metrics = eval_df[eval_df["model"] == "Ensemble"].iloc[0].to_dict()
    logger.info(
        "Ensemble  |  RMSLE=%.5f  R2=%.4f  MAPE=%.2f%%",
        best_metrics["rmsle"], best_metrics["r2"], best_metrics["mape"]
    )

    # ── Step 16: SHAP ─────────────────────────────────────────
    shap_result = None
    if cfg["compute_shap"]:
        logger.info("\n[STEP 16] Computing SHAP ...")
        shap_result = compute_shap(
            lgbm_model, X_val,
            feature_names=feature_cols,
            max_samples=cfg["shap_samples"]
        )
    else:
        logger.info("\n[STEP 16] SHAP skipped")

    # ── Step 17: PSI drift report ─────────────────────────────
    logger.info("\n[STEP 17] PSI drift report ...")
    drift_df   = compute_drift_report(df_train, df_val, feature_cols, top_n=15)
    drift_path = os.path.join(MODEL_DIR, "feature_drift_report.csv")
    drift_df.to_csv(drift_path, index=False)
    logger.info("Drift report → %s", drift_path)

    # ── Step 18: Forecast engine ──────────────────────────────
    logger.info("\n[STEP 18] Running forecast engine ...")
    df_scored = run_forecast_engine(
        df_forecast    = df_val,
        predictions    = ensemble_preds,
        train_end_date = train_end_date,
        encoding_stats = encoding_stats,
    )

    # ── Step 19: Save everything ──────────────────────────────
    logger.info("\n[STEP 19] Saving models and artifacts ...")

    lgbm_path    = save_lgbm_model(lgbm_model, model_name="LightGBM", version="v1")
    tree_path    = save_preprocessor(tree_pre,  name="tree_preprocessor")
    enc_path     = save_encoding_stats(encoding_stats)
    ens_path     = save_ensemble_config(ensemble_weights)
    monitor_path = save_monitor_scores(df_scored, ensemble_preds)

    results_path = os.path.join(MODEL_DIR, "model_experiment_results.csv")
    eval_df.to_csv(results_path, index=False)

    prophet_path = ""
    if prophet_models:
        prophet_path = save_prophet_models(prophet_models)

    lstm_path = lstm_pre_path = ""
    if lstm_model is not None:
        lstm_path     = save_lstm_model(lstm_model)
        lstm_pre_path = save_preprocessor(lstm_pre, name="lstm_preprocessor")

    # ── Step 20: Model card + registry + MLflow ───────────────
    logger.info("\n[STEP 20] Model card + registry + MLflow ...")
    card      = build_model_card(
        eval_df, ensemble_weights, fi_df, shap_result,
        len(df_train), len(df_val), feature_cols, drift_df
    )
    card_path     = save_model_card(card)
    registry_path = update_model_registry(
        lgbm_path, tree_path, ensemble_weights,
        best_metrics, model_card_path=card_path
    )
    mlflow_log_run(
        cfg["mlflow_run_name"], eval_df, ensemble_weights,
        lgbm_model, card, X_train[:100], feature_cols
    )

    elapsed = time.time() - start
    logger.info("\n" + "=" * 65)
    logger.info("PIPELINE COMPLETE  |  time=%.1fs", elapsed)
    logger.info("  Ensemble RMSLE : %.5f", best_metrics["rmsle"])
    logger.info("  Ensemble R2    : %.4f", best_metrics["r2"])
    logger.info("  Ensemble MAPE  : %.2f%%", best_metrics["mape"])
    logger.info("=" * 65)

    return {
        "eval_df":          eval_df,
        "ensemble_weights": ensemble_weights,
        "best_metrics":     best_metrics,
        "lgbm_path":        lgbm_path,
        "prophet_path":     prophet_path,
        "lstm_path":        lstm_path,
        "card_path":        card_path,
        "registry_path":    registry_path,
        "drift_df":         drift_df,
        "feature_cols":     feature_cols,
        "fi_df":            fi_df,
        "elapsed_seconds":  elapsed,
    }


if __name__ == "__main__":
    results = run_training()
    print(f"\n✅ Training complete!")
    print(f"   Ensemble RMSLE : {results['best_metrics']['rmsle']:.5f}")
    print(f"   Model saved    : {results['lgbm_path']}")