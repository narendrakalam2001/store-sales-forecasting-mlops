# ============================================================
# PREPROCESSING — Store Sales Forecasting System
# ============================================================
# Two preprocessor paths (same philosophy as Credit Risk project):
#
#   TREE PATH   (LightGBM, XGBoost):
#       → SmartImputer → Clipper only
#       → NO scaling — trees are scale-invariant
#       → Skewness irrelevant for tree splits
#
#   DISTANCE PATH (LSTM inputs):
#       → SmartImputer → Clipper → log1p (if skew > 0.8) → StandardScaler
#       → Cyclical features: passthrough (already in [-1, 1])
#       → Binary flags: passthrough
#
# IMPUTATION RULES (from EDA):
#   Continuous + no outliers  → MEAN
#   Continuous + has outliers → MEDIAN
#   Categorical               → MODE  (handled in data_loader)
#   Time series gap           → ffill / interpolate (handled in data_loader)
#
# Note: lag/rolling NaNs at series start → drop rows where
#       sales_lag_28 is NaN (first 28 days per store-family)
# ============================================================

import numpy as np
import pandas as pd
import logging

from typing import List, Tuple, Dict, Optional

from sklearn.base          import BaseEstimator, TransformerMixin
from sklearn.compose       import ColumnTransformer
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CLIP_FOLD

logger = logging.getLogger(__name__)


# ============================================================
# CLIPPER — IQR-based outlier clipping (same as Credit Risk)
# ============================================================

class Clipper(BaseEstimator, TransformerMixin):
    """
    Clips feature values to [Q1 - fold*IQR, Q3 + fold*IQR].

    Why needed for time series:
        - Sales lag features can have extreme values (Christmas spike)
        - Clipping prevents extreme rolling-mean features from
          distorting LSTM gradients and tree splits
        - Fitted on train, applied to test — no leakage

    get_feature_names_out() implemented for ColumnTransformer
    compatibility (prevents f0, f1 ... naming issue).
    """

    def __init__(self, fold: float = CLIP_FOLD):
        self.fold = fold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        q1  = np.quantile(X, 0.25, axis=0)
        q3  = np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1

        self.lower_ = q1 - self.fold * iqr
        self.upper_ = q3 + self.fold * iqr

        # Prevent zero-width clip range
        eps         = 1e-9
        self.upper_ = np.where(
            self.upper_ == self.lower_,
            self.upper_ + eps,
            self.upper_
        )
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        n = getattr(self, "n_features_in_", 1)
        return np.array([f"x{i}" for i in range(n)], dtype=object)


# ============================================================
# SMART IMPUTER — mean vs median based on outlier presence
# ============================================================

class SmartImputer(BaseEstimator, TransformerMixin):
    """
    Per-column imputation rule from EDA:
        - Column has IQR outliers → MEDIAN imputation
        - No outliers             → MEAN imputation

    Fitted on train only → applied to train + test (no leakage).
    Handles residual NaN values in numeric feature columns.

    Note: Oil and transactions NaNs are already handled in
    data_loader via interpolate/ffill — this catches any
    remaining edge cases.
    """

    def __init__(self, fold: float = CLIP_FOLD):
        self.fold = fold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.fill_values_ = []
        for j in range(X.shape[1]):
            col       = X[:, j]
            col_clean = col[~np.isnan(col)]

            if len(col_clean) == 0:
                self.fill_values_.append(0.0)
                continue

            q1, q3  = np.quantile(col_clean, [0.25, 0.75])
            iqr     = q3 - q1
            has_out = bool(
                (col_clean < q1 - self.fold * iqr).any() or
                (col_clean > q3 + self.fold * iqr).any()
            )
            # MEDIAN if outliers present, else MEAN
            fill = float(np.median(col_clean) if has_out else np.mean(col_clean))
            self.fill_values_.append(fill)

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        for j, fill in enumerate(self.fill_values_):
            X[np.isnan(X[:, j]), j] = fill
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        n = getattr(self, "n_features_in_", 1)
        return np.array([f"x{i}" for i in range(n)], dtype=object)


# ============================================================
# LOG1P TRANSFORMER (for LSTM skewed features)
# ============================================================

class Log1pTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log1p to skewed continuous features before scaling.
    Clips negatives to 0 first (sales lags can't be negative).
    Used only in the LSTM preprocessor path.
    """

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1] if np.asarray(X).ndim > 1 else 1
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X = np.clip(X, 0, None)
        return np.log1p(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        return np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


# ============================================================
# FEATURE COLUMN CLASSIFIER
# ============================================================

def classify_feature_columns(
    df: pd.DataFrame,
    feature_cols: list,
    skew_threshold: float = 0.8,
) -> Dict[str, List[str]]:
    """
    Classifies feature columns into groups:
        binary   : 0/1 flags (is_weekend, is_promoted, etc.)
        cyclical : sin/cos columns — already in [-1, 1]
        skewed   : continuous, |skew| > threshold → log1p for LSTM
        normal   : continuous, |skew| <= threshold
        ordinal  : low-cardinality ints (month, cluster, store_type_enc)

    These groups drive ColumnTransformer construction.
    Tree models only need skewed + normal + ordinal (Clipper + Impute).
    LSTM also needs log1p on skewed and StandardScaler on all.
    """
    from scipy.stats import skew as scipy_skew

    binary_cols   = []
    cyclical_cols = []
    skewed_cols   = []
    normal_cols   = []
    ordinal_cols  = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        series   = df[col].dropna()
        n_unique = series.nunique()

        # Binary flag
        if set(series.unique()).issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)
            continue

        # Cyclical sin/cos — already normalized
        if col.endswith("_sin") or col.endswith("_cos"):
            cyclical_cols.append(col)
            continue

        # Low-cardinality ordinal integers
        if n_unique <= 20 and (np.issubdtype(series.dtype, np.integer) or series.dtype in [np.int32, np.int64, np.int8, np.int16]):
            ordinal_cols.append(col)
            continue

        # Continuous — check skewness
        try:
            col_skew = float(scipy_skew(series))
        except Exception:
            col_skew = 0.0

        if abs(col_skew) > skew_threshold:
            skewed_cols.append(col)
        else:
            normal_cols.append(col)

    result = {
        "binary":   binary_cols,
        "cyclical": cyclical_cols,
        "skewed":   skewed_cols,
        "normal":   normal_cols,
        "ordinal":  ordinal_cols,
    }

    logger.info(
        "Feature groups  |  binary=%d  cyclical=%d  skewed=%d  normal=%d  ordinal=%d",
        len(binary_cols), len(cyclical_cols),
        len(skewed_cols), len(normal_cols), len(ordinal_cols)
    )
    return result


# ============================================================
# BUILD TREE PREPROCESSOR (LightGBM path)
# ============================================================

def build_tree_preprocessor(
    col_groups: Dict[str, List[str]],
) -> ColumnTransformer:
    """
    LightGBM path:
        Numeric (skewed + normal + ordinal) → SmartImputer → Clipper
        Binary + Cyclical                   → passthrough

    No scaling, no log1p — trees are scale and skew invariant.
    """
    numeric_cols     = col_groups["skewed"] + col_groups["normal"] + col_groups["ordinal"]
    passthrough_cols = col_groups["binary"] + col_groups["cyclical"]

    transformers = []

    if numeric_cols:
        transformers.append((
            "clip_impute",
            Pipeline([
                ("impute", SmartImputer()),
                ("clip",   Clipper()),
            ]),
            numeric_cols
        ))

    if passthrough_cols:
        transformers.append((
            "passthrough",
            "passthrough",
            passthrough_cols
        ))

    preprocessor = ColumnTransformer(
        transformers              = transformers,
        remainder                 = "drop",
        verbose_feature_names_out = False,
    )

    logger.info(
        "Tree preprocessor  |  numeric=%d  passthrough=%d",
        len(numeric_cols), len(passthrough_cols)
    )
    return preprocessor


# ============================================================
# BUILD LSTM PREPROCESSOR (neural network path)
# ============================================================

def build_lstm_preprocessor(
    col_groups: Dict[str, List[str]],
) -> ColumnTransformer:
    """
    LSTM path:
        Skewed continuous → SmartImputer → Clipper → Log1p → StandardScaler
        Normal continuous → SmartImputer → Clipper → StandardScaler
        Ordinal integers  → SmartImputer → StandardScaler
        Binary / cyclical → passthrough

    Why log1p for skewed:
        LSTM uses gradient descent — large variance inputs slow convergence.
        Sales lag features inherit heavy right skew (Christmas spikes).
        log1p compresses range AND reduces outlier influence on gradients.
    """
    transformers = []

    if col_groups["skewed"]:
        transformers.append((
            "skewed",
            Pipeline([
                ("impute", SmartImputer()),
                ("clip",   Clipper()),
                ("log1p",  Log1pTransformer()),
                ("scale",  StandardScaler()),
            ]),
            col_groups["skewed"]
        ))

    if col_groups["normal"]:
        transformers.append((
            "normal",
            Pipeline([
                ("impute", SmartImputer()),
                ("clip",   Clipper()),
                ("scale",  StandardScaler()),
            ]),
            col_groups["normal"]
        ))

    if col_groups["ordinal"]:
        transformers.append((
            "ordinal",
            Pipeline([
                ("impute", SmartImputer()),
                ("scale",  StandardScaler()),
            ]),
            col_groups["ordinal"]
        ))

    passthrough_cols = col_groups["binary"] + col_groups["cyclical"]
    if passthrough_cols:
        transformers.append((
            "passthrough",
            "passthrough",
            passthrough_cols
        ))

    preprocessor = ColumnTransformer(
        transformers              = transformers,
        remainder                 = "drop",
        verbose_feature_names_out = False,
    )

    logger.info(
        "LSTM preprocessor  |  skewed=%d  normal=%d  ordinal=%d  passthrough=%d",
        len(col_groups["skewed"]), len(col_groups["normal"]),
        len(col_groups["ordinal"]), len(passthrough_cols)
    )
    return preprocessor


# ============================================================
# DROP NaN ROWS (from lag features)
# ============================================================

def drop_lag_nans(df: pd.DataFrame, lag_col: str = "sales_lag_28") -> pd.DataFrame:
    """
    Removes rows where the longest lag feature is NaN.

    Why: first 28 days of each (store, family) series have NaN lag_28.
    These rows have incomplete input features → cannot train on them.

    Impact: drops ~28 x 54 x 33 = ~49,896 rows from ~3M → negligible.
    If lag_28 is valid, all shorter lags (1,7,14) are also valid.
    """
    before = len(df)
    if lag_col in df.columns:
        df = df.dropna(subset=[lag_col]).reset_index(drop=True)
    after = len(df)
    logger.info(
        "Dropped %d NaN-lag rows (%s)  |  %d → %d",
        before - after, lag_col, before, after
    )
    return df


# ============================================================
# PREPARE LGBM ARRAYS
# ============================================================

def prepare_lgbm_arrays(
    df_train:     pd.DataFrame,
    df_val:       pd.DataFrame,
    feature_cols: list,
    preprocessor: ColumnTransformer,
) -> Tuple:
    """
    Fits tree preprocessor on train, transforms train + val.

    Target: df["target"] = log1p(sales) — set in feature_engineering.py.

    Returns: (X_train, y_train, X_val, y_val, fitted_preprocessor)
    """
    X_train_raw = df_train[feature_cols]
    X_val_raw   = df_val[feature_cols]
    y_train     = df_train["target"].values
    y_val       = df_val["target"].values

    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_val   = preprocessor.transform(X_val_raw)

    logger.info(
        "LGBM arrays  |  X_train=%s  X_val=%s  y mean=%.4f",
        X_train.shape, X_val.shape, y_train.mean()
    )
    return X_train, y_train, X_val, y_val, preprocessor


# ============================================================
# PREPARE LSTM SEQUENCES
# ============================================================

def prepare_lstm_sequences(
    df:           pd.DataFrame,
    feature_cols: list,
    seq_len:      int,
    preprocessor: ColumnTransformer,
    is_train:     bool = True,
) -> Tuple:
    """
    Builds 3D arrays for LSTM: (samples, seq_len, n_features).

    Per (store_nbr, family) group:
        - Creates overlapping windows of length seq_len
        - Target at time t = log1p(sales[t])
        - Input = scaled features at [t-seq_len, ..., t-1]

    shift(1) in lag/rolling already prevents leakage in features.
    Here we just slice the pre-computed feature matrix into windows.

    Returns:
        is_train=True  → (X_seq, y_seq, fitted_preprocessor)
        is_train=False → (X_seq, fitted_preprocessor)
    """
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    if is_train:
        X_scaled = preprocessor.fit_transform(df[feature_cols])
    else:
        X_scaled = preprocessor.transform(df[feature_cols])

    X_seqs, y_seqs = [], []

    for (store, family), grp_idx in df.groupby(["store_nbr", "family"]).groups.items():
        idx_list = list(grp_idx)
        grp_X    = X_scaled[idx_list]
        grp_y    = df.loc[idx_list, "target"].values if is_train else None
        n        = len(grp_X)

        for i in range(seq_len, n):
            X_seqs.append(grp_X[i - seq_len : i])
            if is_train:
                y_seqs.append(grp_y[i])

    if not X_seqs:
        raise ValueError("No LSTM sequences created — check seq_len vs data length")

    X_out = np.array(X_seqs, dtype=np.float32)
    logger.info("LSTM sequences  |  shape=%s", X_out.shape)

    if is_train:
        return X_out, np.array(y_seqs, dtype=np.float32), preprocessor
    return X_out, preprocessor


# ============================================================
# WALK-FORWARD SPLITS
# ============================================================

def walk_forward_splits(
    df:       pd.DataFrame,
    n_splits: int = 3,
    val_days: int = 16,
) -> list:
    """
    Time-series aware train/validation splits.
    NEVER shuffle — respect temporal order.
    Val window = 16 days → matches Kaggle test horizon.

    Returns list of (train_mask, val_mask, val_start, val_end).

    Example n_splits=3, val_days=16:
        Split 1: train=[start .. D-48], val=[D-48 .. D-32]
        Split 2: train=[start .. D-32], val=[D-32 .. D-16]
        Split 3: train=[start .. D-16], val=[D-16 .. D]
    """
    dates   = sorted(df["date"].unique())
    n_dates = len(dates)
    splits  = []

    for i in range(n_splits, 0, -1):
        val_end_idx   = n_dates - (i - 1) * val_days
        val_start_idx = val_end_idx - val_days

        if val_start_idx <= 0:
            logger.warning("Not enough dates for split %d — skipping", i)
            continue

        val_start = dates[val_start_idx]
        val_end   = dates[min(val_end_idx, n_dates - 1)]

        train_mask = df["date"] < val_start
        val_mask   = (df["date"] >= val_start) & (df["date"] <= val_end)

        splits.append((train_mask, val_mask, val_start, val_end))

        logger.info(
            "Split %d/%d  |  val=[%s → %s]  train=%d  val=%d",
            n_splits - i + 1, n_splits,
            val_start, val_end,
            train_mask.sum(), val_mask.sum()
        )

    return splits