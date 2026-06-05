# ============================================================
# FEATURE ENGINEERING — Store Sales Forecasting System
# ============================================================
# All features derived from EDA findings:
#
# FEATURE GROUPS:
#   1. Lag features         — sales_lag_1, 7, 14, 28
#   2. Rolling features     — rolling_mean/std — 7d, 14d, 28d
#   3. Promotion features   — onpromotion lags + rolling
#   4. Calendar features    — day, week, month, year, quarter
#   5. Cyclical encoding    — sin/cos (LSTM only)
#   6. Oil features         — raw + rolling smoothed
#   7. Transaction features — txn_lag_1, txn_lag_7, txn_rolling_7d
#   8. Store metadata       — type, cluster, city target-encoded
#   9. Family encoding      — label encode + target encode
#  10. Interaction features — promo × family, oil × onpromotion
#
# DESIGN RULES:
#   - All lag/rolling computed WITHIN each (store_nbr, family) group
#   - shift(1) before rolling → prevent target leakage
#   - log1p(sales) is the TRAINING TARGET (not a feature)
#   - Tree models: no scaling needed
#   - LSTM: cyclical + StandardScaler applied separately in preprocessing.py
# ============================================================

import numpy as np
import pandas as pd
import logging

from src.config import (
    LAG_DAYS, ROLLING_WINDOWS, PROMO_LAG_DAYS,
    LOG_TARGET, RANDOM_STATE
)

logger = logging.getLogger(__name__)


# ============================================================
# CALENDAR FEATURES
# ============================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts temporal signal from the date column.
    EDA showed: month, day_of_week, year are top correlated with sales.
    """
    df = df.copy()

    df["day_of_week"]  = df["date"].dt.dayofweek          # 0=Mon, 6=Sun
    df["day_of_month"] = df["date"].dt.day
    df["month"]        = df["date"].dt.month
    df["year"]         = df["date"].dt.year
    df["quarter"]      = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["date"].dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)

    # ── Days since start of training (trend feature) ──────────
    min_date = df["date"].min()
    df["days_since_start"] = (df["date"] - min_date).dt.days

    # ── Payday proxy (1st and 15th of month → higher grocery sales) ──
    df["is_payday"] = df["date"].dt.day.isin([1, 15]).astype(int)

    logger.debug("Calendar features added: %d", df.shape[1])
    return df


# ============================================================
# CYCLICAL ENCODING (for LSTM)
# ============================================================

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sine/cosine encoding of periodic features.
    Needed for LSTM — treats Mon and Sun as "close" not "far apart".
    LightGBM doesn't need this — trees split on raw values fine.
    """
    df = df.copy()

    df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"]    = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"]    = np.cos(2 * np.pi * df["week_of_year"] / 52)

    return df


# ============================================================
# LAG FEATURES (per store × family group)
# ============================================================

def add_lag_features(df: pd.DataFrame, lag_days: list = LAG_DAYS) -> pd.DataFrame:
    """
    Sales lag features: what was the sales 1, 7, 14, 28 days ago?

    Critical design:
        - Computed WITHIN each (store_nbr, family) group
        - Prevents lag "leaking" across stores/families
        - EDA ACF confirmed strong autocorrelation at lags 1, 7, 14, 28

    After adding lags, rows with NaN lag values are expected at the
    start of each series — handled during train/test split (drop or fill).
    """
    df = df.copy()
    df = df.sort_values(["store_nbr", "family", "date"])

    grp = df.groupby(["store_nbr", "family"])["sales"]

    for lag in lag_days:
        df[f"sales_lag_{lag}"] = grp.shift(lag)

    # Log1p lag features (reduces impact of outlier days)
    for lag in lag_days:
        col = f"sales_lag_{lag}"
        df[f"log1p_lag_{lag}"] = np.log1p(df[col].clip(lower=0))

    logger.debug("Lag features added: lags=%s", lag_days)
    return df


# ============================================================
# ROLLING FEATURES (per store × family group)
# ============================================================

def add_rolling_features(df: pd.DataFrame, windows: list = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Rolling mean and std of sales.

    Design:
        - shift(1) BEFORE rolling → no target leakage
          (rolling uses values from t-window to t-1, NOT t)
        - min_periods=1 prevents NaN for short series

    EDA finding: rolling_mean_7d and rolling_mean_28d are top
    autocorrelated features with current sales.
    """
    df = df.copy()
    df = df.sort_values(["store_nbr", "family", "date"])

    grp = df.groupby(["store_nbr", "family"])["sales"]

    for w in windows:
        shifted = grp.shift(1)   # shift(1) → look-back only (no leakage)
        df[f"rolling_mean_{w}d"] = shifted.transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}d"] = shifted.transform(
            lambda x: x.rolling(w, min_periods=1).std().fillna(0)
        )

    # ── Exponential weighted mean (captures recent trend better) ──
    df["ewm_alpha_07"] = grp.shift(1).transform(
        lambda x: x.ewm(alpha=0.7, adjust=False).mean()
    )
    df["ewm_alpha_03"] = grp.shift(1).transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )

    logger.debug("Rolling features added: windows=%s", windows)
    return df


# ============================================================
# PROMOTION FEATURES
# ============================================================

def add_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Promotion signal features.

    EDA finding:
        - Mann-Whitney U test: promoted rows have significantly higher sales
        - Promotion lift is highest for GROCERY I, BEVERAGES, CLEANING families
        - Promo lag: effect of promotion from previous week

    Note: onpromotion is available in test.csv → no leakage risk.
    """
    df = df.copy()
    df = df.sort_values(["store_nbr", "family", "date"])

    grp_promo = df.groupby(["store_nbr", "family"])["onpromotion"]

    # Binary flag
    df["is_promoted"] = (df["onpromotion"] > 0).astype(int)

    # Promotion lag features
    for lag in PROMO_LAG_DAYS:
        df[f"promo_lag_{lag}"] = grp_promo.shift(lag).fillna(0)

    # Rolling promotion count
    df["promo_rolling_7d"] = grp_promo.shift(1).transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    # Promo × log_sales interaction (captured at lag — no leakage)
    df["promo_x_lag7"] = df["is_promoted"] * df.get("sales_lag_7", 0)

    logger.debug("Promotion features added")
    return df


# ============================================================
# OIL FEATURES
# ============================================================

def add_oil_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oil price features.

    EDA finding:
        - Ecuador is oil-dependent → oil crash 2015 affected consumer spending
        - Negative correlation with total sales
        - Rolling mean smooths out daily noise

    Missing oil values already handled in data_loader (interpolated).
    """
    df = df.copy()

    # Rolling oil price (macro trend signal)
    oil_sorted = df.sort_values("date")
    for w in [7, 28]:
        df[f"oil_rolling_{w}d"] = (
            df.sort_values("date")
              .groupby("date")["dcoilwtico"]
              .transform("first")
              .rolling(w, min_periods=1)
              .mean()
              .values
        )

    # Oil price change rate
    daily_oil = df[["date", "dcoilwtico"]].drop_duplicates("date").sort_values("date")
    daily_oil["oil_pct_change_7d"] = daily_oil["dcoilwtico"].pct_change(7).fillna(0)
    df = df.merge(daily_oil[["date", "oil_pct_change_7d"]], on="date", how="left")

    # Oil regime: high vs low (above/below 80 USD = rough pre/post crash)
    df["oil_regime_high"] = (df["dcoilwtico"] >= 80).astype(int)

    logger.debug("Oil features added")
    return df


# ============================================================
# TRANSACTION FEATURES
# ============================================================

def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transactions = store-level footfall proxy.
    Available in training data but NOT in test.csv directly.
    → Use lag features only (safe for both train and test).

    EDA finding: txn-sales correlation = 0.83 (very strong).
    """
    df = df.copy()
    df = df.sort_values(["store_nbr", "date"])

    grp_txn = df.groupby("store_nbr")["transactions"]

    # Lag features (safe — available at prediction time via history)
    df["txn_lag_1"]       = grp_txn.shift(1)
    df["txn_lag_7"]       = grp_txn.shift(7)
    df["txn_rolling_7d"]  = grp_txn.shift(1).transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df["txn_rolling_28d"] = grp_txn.shift(1).transform(
        lambda x: x.rolling(28, min_periods=1).mean()
    )

    # Fill any remaining nulls with store-level mean
    for col in ["txn_lag_1", "txn_lag_7", "txn_rolling_7d", "txn_rolling_28d"]:
        df[col] = df.groupby("store_nbr")[col].transform(
            lambda x: x.fillna(x.mean())
        )

    logger.debug("Transaction features added")
    return df


# ============================================================
# STORE & FAMILY ENCODING
# ============================================================

def add_store_family_features(df: pd.DataFrame, is_train: bool = True,
                               encoding_stats: dict = None) -> tuple:
    """
    Store and family encoding.

    store_type_enc: already done in data_loader (A=4 ... E=0)
    cluster:        ordinal integer (1–17) — use directly for trees

    Family target encoding:
        family_sales_mean = mean(log1p(sales)) per family on train set
        → Applied to both train and test (using train stats)

    Store target encoding:
        store_sales_mean = mean(log1p(sales)) per store on train set

    Returns: (df, encoding_stats) — stats saved for test-time reuse.
    """
    df = df.copy()

    if is_train:
        # ── Compute encoding stats from train ──────────────────
        df["log1p_sales"] = np.log1p(df["sales"])

        family_mean = df.groupby("family")["log1p_sales"].mean().to_dict()
        store_mean  = df.groupby("store_nbr")["log1p_sales"].mean().to_dict()
        store_cluster_mean = df.groupby("cluster")["log1p_sales"].mean().to_dict()

        encoding_stats = {
            "family_mean":         family_mean,
            "store_mean":          store_mean,
            "store_cluster_mean":  store_cluster_mean,
            "global_mean":         df["log1p_sales"].mean(),
        }

        df.drop(columns=["log1p_sales"], inplace=True)
    else:
        if encoding_stats is None:
            raise ValueError("encoding_stats required for test-time encoding")

    # ── Apply encodings ────────────────────────────────────────
    global_mean = encoding_stats["global_mean"]

    df["family_sales_mean"] = df["family"].map(
        encoding_stats["family_mean"]
    ).fillna(global_mean)

    df["store_sales_mean"] = df["store_nbr"].map(
        encoding_stats["store_mean"]
    ).fillna(global_mean)

    df["cluster_sales_mean"] = df["cluster"].map(
        encoding_stats["store_cluster_mean"]
    ).fillna(global_mean)

    # ── Label encode family ────────────────────────────────────
    if is_train:
        families = sorted(df["family"].unique())
        family_label_map = {f: i for i, f in enumerate(families)}
        encoding_stats["family_label_map"] = family_label_map

    df["family_enc"] = df["family"].map(
        encoding_stats["family_label_map"]
    ).fillna(-1).astype(int)

    # ── Zero-inflation flag per family ────────────────────────
    if is_train:
        zero_pct = df.groupby("family")["sales"].apply(
            lambda x: (x == 0).mean()
        ).to_dict()
        encoding_stats["family_zero_pct"] = zero_pct

    df["family_zero_pct"] = df["family"].map(
        encoding_stats.get("family_zero_pct", {})
    ).fillna(0.0)

    logger.debug("Store & family encoding done")
    return df, encoding_stats


# ============================================================
# MASTER FEATURE ENGINEERING FUNCTION
# ============================================================

def build_features(
    df: pd.DataFrame,
    is_train: bool = True,
    encoding_stats: dict = None,
    add_cyclical: bool = False,
) -> tuple:
    """
    Master function — runs all feature engineering steps in order.

    Args:
        df             : merged DataFrame from data_loader.build_merged_dataset()
        is_train       : True = compute encoding stats, False = reuse existing
        encoding_stats : required if is_train=False
        add_cyclical   : True = add sin/cos features (for LSTM)

    Returns:
        (df_featured, encoding_stats)
    """
    logger.info("Starting feature engineering  |  shape=%s  is_train=%s", df.shape, is_train)

    # ── Step 1: Calendar ──────────────────────────────────────
    df = add_calendar_features(df)

    # ── Step 2: Cyclical (optional — LSTM only) ───────────────
    if add_cyclical:
        df = add_cyclical_features(df)

    # ── Step 3: Lag features ──────────────────────────────────
    df = add_lag_features(df)

    # ── Step 4: Rolling features ──────────────────────────────
    df = add_rolling_features(df)

    # ── Step 5: Promotion features ────────────────────────────
    df = add_promotion_features(df)

    # ── Step 6: Oil features ──────────────────────────────────
    df = add_oil_features(df)

    # ── Step 7: Transaction features ─────────────────────────
    df = add_transaction_features(df)

    # ── Step 8: Store & family encoding ──────────────────────
    df, encoding_stats = add_store_family_features(df, is_train, encoding_stats)

    # ── Step 9: Target column ─────────────────────────────────
    if is_train and "sales" in df.columns:
        df["target"] = np.log1p(df["sales"].clip(lower=0))  # RMSLE → log1p target

    logger.info(
        "Feature engineering done  |  shape=%s  |  total_features=%d",
        df.shape, df.shape[1]
    )

    return df, encoding_stats


# ============================================================
# GET FEATURE COLUMNS FOR MODEL
# ============================================================

def get_feature_columns(df: pd.DataFrame, exclude: list = None) -> list:
    """
    Returns the final list of feature columns for model training.
    Excludes: id, date, sales, target, family (string), city, state, store_type (string).
    """
    # Columns to always exclude
    always_exclude = {
        "id", "date", "sales", "target",
        "family",           # use family_enc instead
        "city",             # high cardinality — use store_sales_mean instead
        "state",            # high cardinality — use store_sales_mean instead
        "store_type",       # use store_type_enc instead
        "description",      # holiday description — too high cardinality
        "locale",           # use is_national/regional/local instead
        "locale_name",      # high cardinality
        "transferred",      # use is_transferred instead
        "type",             # holiday type raw — already encoded
        "transactions",     # raw txn — NOT in test.csv → use txn_lag_* instead
    }

    if exclude:
        always_exclude.update(exclude)

    feature_cols = [
        c for c in df.columns
        if c not in always_exclude
        and df[c].dtype != object
        and df[c].dtype.name != "category"
    ]

    logger.info("Feature columns: %d selected", len(feature_cols))
    return feature_cols