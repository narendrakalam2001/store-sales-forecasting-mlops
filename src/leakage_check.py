# ============================================================
# LEAKAGE CHECK — Store Sales Forecasting System
# ============================================================
# Time series leakage is different from tabular leakage:
#
#   TABULAR LEAKAGE  : feature = target (direct or near-perfect corr)
#   TIME SERIES LEAKAGE:
#       1. Future data used to compute past features (look-ahead)
#       2. Target included in features without proper shift
#       3. Aggregates computed on full dataset before split
#       4. External columns (transactions) used without lag
#
# We check:
#   - Any column with correlation >= threshold to target
#   - Any lag/rolling column missing shift(1) (detects look-ahead)
#   - 'transactions' used without lag suffix (direct leakage risk)
#   - 'sales' column present in features (direct target leakage)
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def detect_leakage(
    X_train:        pd.DataFrame,
    y_train:        pd.Series,
    threshold_corr: float = 0.98,
) -> list:
    """
    Detects potential data leakage in time series features.

    Checks:
        1. Direct target match (feature == target)
        2. Near-perfect correlation with target (>= threshold)
        3. 'sales' column present without lag suffix (target leak)
        4. 'transactions' column without lag/rolling suffix (future leak)
        5. Any rolling/ewm column without shift (look-ahead)

    Args:
        X_train        : feature DataFrame (after split)
        y_train        : log1p(sales) target Series
        threshold_corr : correlation threshold for flagging

    Returns:
        List of warning strings. Empty = no leakage detected.
    """
    warnings = []

    for col in X_train.columns:

        # ── Check 1: Direct target match ──────────────────────
        try:
            if X_train[col].equals(y_train.astype(X_train[col].dtype)):
                warnings.append(
                    f"[LEAKAGE] '{col}' is identical to target → remove"
                )
                continue
        except Exception:
            pass

        # ── Check 2: Near-perfect correlation ─────────────────
        if np.issubdtype(X_train[col].dtype, np.number):
            try:
                corr = abs(np.corrcoef(
                    X_train[col].fillna(0),
                    y_train.fillna(0)
                )[0, 1])
                if corr >= threshold_corr:
                    warnings.append(
                        f"[LEAKAGE] '{col}' corr={corr:.4f} >= {threshold_corr} "
                        f"with target → possible look-ahead"
                    )
            except Exception as e:
                logger.debug("Corr check failed for %s: %s", col, e)

    # ── Check 3: Raw 'sales' in features ──────────────────────
    if "sales" in X_train.columns:
        warnings.append(
            "[LEAKAGE] 'sales' column present in features — "
            "this IS the target → drop immediately"
        )

    # ── Check 4: 'transactions' without lag suffix ────────────
    # Only flag if actually in feature columns (get_feature_columns excludes it)
    if "transactions" in X_train.columns:
        warnings.append(
            "[LEAKAGE] 'transactions' in features directly — "
            "NOT available in test.csv → use txn_lag_1, txn_lag_7, txn_rolling_7d"
        )
    else:
        logger.info(
            "transactions: correctly excluded from features ✅ "
            "(txn_lag_* used instead)"
        )

    # ── Check 5: Rolling/EWM without lag ─────────────────────
    # These column names suggest rolling was computed without shift
    suspect_patterns = ["rolling_sales", "ewm_sales", "expanding_sales"]
    for col in X_train.columns:
        for pattern in suspect_patterns:
            if pattern in col and "lag" not in col:
                warnings.append(
                    f"[LEAKAGE] '{col}' looks like rolling on raw sales "
                    f"without shift → ensure shift(1) was applied before rolling"
                )

    # ── Log results ───────────────────────────────────────────
    if warnings:
        logger.warning("Leakage check found %d issue(s):", len(warnings))
        for w in warnings:
            logger.warning("  %s", w)
    else:
        logger.info("Leakage check passed — no obvious leakage detected ✅")

    return warnings