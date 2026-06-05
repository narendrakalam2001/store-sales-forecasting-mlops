# ============================================================
# CONFIGURATION — Store Sales Forecasting System
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE  = 42
N_JOBS        = -1

# ── Data paths ───────────────────────────────────────────────
DATA_DIR = r"D:\Data Science Datasets\store-sales-time-series-forecasting"

TRAIN_PATH        = os.path.join(DATA_DIR, "train.csv")
TEST_PATH         = os.path.join(DATA_DIR, "test.csv")
STORES_PATH       = os.path.join(DATA_DIR, "stores.csv")
OIL_PATH          = os.path.join(DATA_DIR, "oil.csv")
HOLIDAYS_PATH     = os.path.join(DATA_DIR, "holidays_events.csv")
TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions.csv")

# ── Output directories ────────────────────────────────────────
MODEL_DIR  = "forecast_models"
LOG_DIR    = "logs"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Time series config ────────────────────────────────────────
FORECAST_HORIZON  = 16          # Kaggle test = 16 days ahead
TRAIN_CUTOFF_DAYS = 365         # Last 365 days used for validation
LAG_DAYS          = [1, 7, 14, 28]      # Sales lag features
ROLLING_WINDOWS   = [7, 14, 28]         # Rolling mean/std windows
PROMO_LAG_DAYS    = [1, 7]              # Promotion lag features

# ── Target config ─────────────────────────────────────────────
# RMSLE metric → train on log1p(sales), back-transform with expm1()
TARGET_COL        = "sales"
LOG_TARGET        = True        # always log1p transform target

# ── Outlier clipping ─────────────────────────────────────────
CLIP_FOLD = 1.5                 # IQR multiplier for Clipper

# ── Feature selection ─────────────────────────────────────────
# Time series: no SelectKBest — LightGBM uses all features
# LSTM: top N features by LightGBM importance
LSTM_TOP_FEATURES = 20

# ── LightGBM defaults ─────────────────────────────────────────
LGBM_PARAMS = {
    "objective":     "regression",
    "metric":        "rmse",           # RMSE on log target ≈ RMSLE on raw
    "n_estimators":  2000,
    "learning_rate": 0.05,
    "num_leaves":    63,
    "max_depth":     -1,
    "subsample":     0.8,
    "colsample_bytree": 0.8,
    "random_state":  RANDOM_STATE,
    "n_jobs":        N_JOBS,
    "verbose":       -1,
}

# ── Prophet defaults ──────────────────────────────────────────
PROPHET_PARAMS = {
    "yearly_seasonality":  True,
    "weekly_seasonality":  True,
    "daily_seasonality":   False,
    "seasonality_mode":    "multiplicative",   # retail = multiplicative
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
}

# ── LSTM config ───────────────────────────────────────────────
LSTM_SEQ_LEN    = 28           # input window — 28 days of history
LSTM_BATCH_SIZE = 512
LSTM_EPOCHS     = 30
LSTM_PATIENCE   = 5            # early stopping patience

# ── Ensemble weights (Optuna will tune these) ─────────────────
# Defaults before tuning:
ENSEMBLE_WEIGHTS = {
    "lgbm":    0.60,
    "prophet": 0.25,
    "lstm":    0.15,
}

# ── Drift Monitoring Thresholds (PSI) ─────────────────────────
# PSI < 0.10   → No action needed (stable)
# PSI 0.10–0.20 → Moderate shift  — monitor closely
# PSI > 0.20   → Critical drift   — retrain recommended
PSI_MODERATE = 0.10
PSI_HIGH     = 0.20

# ── Champion vs Challenger Gates ──────────────────────────────
# Gate 1: Challenger must beat champion RMSLE by at least this margin
MIN_RMSLE_IMPROVEMENT = 0.005

# Gate 2: Challenger R² must meet minimum threshold
MIN_R2_THRESHOLD = 0.80

# ── Store & family counts (from EDA) ──────────────────────────
N_STORES   = 54
N_FAMILIES = 33

# ── Earthquake event ─────────────────────────────────────────
EARTHQUAKE_DATE        = "2016-04-16"
EARTHQUAKE_WINDOW_PRE  = 7    # days before
EARTHQUAKE_WINDOW_POST = 30   # days after — recovery period

# ── Store type mapping ────────────────────────────────────────
STORE_TYPE_ORDER = ["A", "B", "C", "D", "E"]  # A = largest, E = smallest