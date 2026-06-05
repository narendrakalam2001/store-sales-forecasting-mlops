# ============================================================
# DATA LOADER — Store Sales Forecasting System
# ============================================================
# Loads, validates, and merges all 6 input files:
#   train.csv + stores.csv + oil.csv + holidays_events.csv
#   + transactions.csv + (test.csv for inference)
#
# Key design decisions from EDA:
#   - oil.dcoilwtico: 43 missing → interpolate(method='time')
#   - transactions: merge on date+store_nbr, missing → ffill per store
#   - holidays: national/regional/local flags separately
#   - sales = 0 on missing (store,family,date) combos → structural zero
#   - earthquake Apr 2016 → binary flag added here
# ============================================================

import os
import numpy as np
import pandas as pd
import logging

from src.config import (
    TRAIN_PATH, TEST_PATH, STORES_PATH, OIL_PATH,
    HOLIDAYS_PATH, TRANSACTIONS_PATH,
    EARTHQUAKE_DATE, EARTHQUAKE_WINDOW_PRE, EARTHQUAKE_WINDOW_POST,
    RANDOM_STATE
)

logger = logging.getLogger(__name__)

# ── Required columns ──────────────────────────────────────────
TRAIN_REQUIRED = ["date", "store_nbr", "family", "sales", "onpromotion"]
TEST_REQUIRED  = ["date", "store_nbr", "family", "onpromotion"]


# ============================================================
# LOAD RAW FILES
# ============================================================

def load_raw_files(
    train_path:        str = TRAIN_PATH,
    test_path:         str = TEST_PATH,
    stores_path:       str = STORES_PATH,
    oil_path:          str = OIL_PATH,
    holidays_path:     str = HOLIDAYS_PATH,
    transactions_path: str = TRANSACTIONS_PATH,
):
    """
    Loads all 6 CSV files with correct dtypes and date parsing.

    Returns:
        dict of {name: DataFrame}
    """
    logger.info("Loading raw files ...")

    train        = pd.read_csv(train_path,        parse_dates=["date"])
    test         = pd.read_csv(test_path,         parse_dates=["date"])
    stores       = pd.read_csv(stores_path)
    oil          = pd.read_csv(oil_path,          parse_dates=["date"])
    holidays     = pd.read_csv(holidays_path,     parse_dates=["date"])
    transactions = pd.read_csv(transactions_path, parse_dates=["date"])

    # ── Normalize column names ────────────────────────────────
    for df in [train, test, stores, oil, holidays, transactions]:
        df.columns = df.columns.str.strip().str.lower()

    logger.info(
        "Loaded  |  train=%s  test=%s  stores=%s  oil=%s  holidays=%s  txn=%s",
        train.shape, test.shape, stores.shape,
        oil.shape, holidays.shape, transactions.shape
    )

    return {
        "train":        train,
        "test":         test,
        "stores":       stores,
        "oil":          oil,
        "holidays":     holidays,
        "transactions": transactions,
    }


# ============================================================
# VALIDATE TRAIN DATA
# ============================================================

def validate_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Schema check + deduplication + basic sanity.
    """
    # ── Required columns ──────────────────────────────────────
    missing_cols = [c for c in TRAIN_REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"train.csv missing columns: {missing_cols}")

    # ── Target range ──────────────────────────────────────────
    if df["sales"].min() < 0:
        n_neg = (df["sales"] < 0).sum()
        logger.warning("%d rows with negative sales — clipping to 0", n_neg)
        df["sales"] = df["sales"].clip(lower=0)

    # ── Duplicate rows ────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["date", "store_nbr", "family"], keep="first")
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate (date, store_nbr, family) rows", dropped)

    # ── Size check ────────────────────────────────────────────
    if len(df) < 1000:
        raise ValueError(f"Training data too small: {len(df)} rows")

    logger.info(
        "Validation passed  |  shape=%s  |  date_range=%s → %s  |  zero_sales=%.1f%%",
        df.shape,
        df["date"].min().date(),
        df["date"].max().date(),
        (df["sales"] == 0).mean() * 100
    )

    return df.reset_index(drop=True)


# ============================================================
# PROCESS OIL — interpolate missing dates
# ============================================================

def process_oil(oil: pd.DataFrame) -> pd.DataFrame:
    full_range = pd.date_range(oil["date"].min(), oil["date"].max(), freq="D")

    # Keep DatetimeIndex during interpolation — method='time' requires it
    oil_indexed = (
        oil.set_index("date")
           .reindex(full_range)
           .rename_axis("date")
    )
    oil_indexed["dcoilwtico"] = (
        oil_indexed["dcoilwtico"]
        .interpolate(method="time")   # DatetimeIndex present here — works correctly
        .bfill()
    )
    oil = oil_indexed.reset_index()   # reset AFTER interpolation

    logger.info(
        "Oil processed  |  missing after fill=%d  |  range=%.2f–%.2f",
        oil["dcoilwtico"].isna().sum(),
        oil["dcoilwtico"].min(),
        oil["dcoilwtico"].max()
    )
    return oil

# ============================================================
# PROCESS HOLIDAYS — create flag columns
# ============================================================

def process_holidays(holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Creates per-date binary flags:
        is_national_holiday  — National locale, not transferred
        is_regional_holiday  — Regional locale
        is_local_holiday     — Local locale
        is_holiday_event     — type=Event (fairs, competitions)
        is_transferred       — transferred holiday
        holiday_type         — raw type string
        holiday_locale       — raw locale string

    Returns one row per date (aggregated — take max for binary flags).
    Some dates have multiple holiday entries (e.g. national + earthquake event).
    """
    h = holidays.copy()

    # ── Type flags ────────────────────────────────────────────
    h["is_national_holiday"] = (
        (h["locale"] == "National") &
        (h["type"].isin(["Holiday", "Transfer", "Bridge"])) &
        (~h["transferred"])
    ).astype(int)

    h["is_regional_holiday"] = (
        (h["locale"] == "Regional") &
        (~h["transferred"])
    ).astype(int)

    h["is_local_holiday"] = (
        (h["locale"] == "Local") &
        (~h["transferred"])
    ).astype(int)

    h["is_holiday_event"] = (h["type"] == "Event").astype(int)
    h["is_transferred"]   = h["transferred"].astype(int)
    h["is_work_day"]      = (h["type"] == "Work Day").astype(int)

    # ── Aggregate to one row per date ─────────────────────────
    # (some dates have multiple holiday entries)
    agg = h.groupby("date").agg(
        is_national_holiday = ("is_national_holiday", "max"),
        is_regional_holiday = ("is_regional_holiday", "max"),
        is_local_holiday    = ("is_local_holiday",    "max"),
        is_holiday_event    = ("is_holiday_event",    "max"),
        is_transferred      = ("is_transferred",      "max"),
        is_work_day         = ("is_work_day",         "max"),
        holiday_count       = ("date",                "count"),   # how many events on this date
    ).reset_index()

    # ── Any holiday flag ──────────────────────────────────────
    agg["is_any_holiday"] = (
        (agg["is_national_holiday"] |
         agg["is_regional_holiday"]  |
         agg["is_local_holiday"])
    ).astype(int)

    logger.info(
        "Holidays processed  |  national=%d  regional=%d  local=%d  events=%d",
        agg["is_national_holiday"].sum(),
        agg["is_regional_holiday"].sum(),
        agg["is_local_holiday"].sum(),
        agg["is_holiday_event"].sum()
    )

    return agg


# ============================================================
# ADD PROXIMITY TO HOLIDAY
# ============================================================

def add_holiday_proximity(df: pd.DataFrame, holidays_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Adds days_to_next_holiday and days_since_last_holiday.
    Demand anticipation: sales spike 1-2 days BEFORE holiday.
    Post-holiday dip: sales drop 1-2 days AFTER holiday.
    """
    national_dates = pd.to_datetime(
        holidays_agg[holidays_agg["is_national_holiday"] == 1]["date"].unique()
    )

    def _days_to_next(date, holiday_dates):
        future = holiday_dates[holiday_dates >= date]
        return (future.min() - date).days if len(future) > 0 else 999

    def _days_since_last(date, holiday_dates):
        past = holiday_dates[holiday_dates <= date]
        return (date - past.max()).days if len(past) > 0 else 999

    unique_dates = df["date"].unique()
    proximity_map = {}
    for d in unique_dates:
        proximity_map[d] = {
            "days_to_holiday":    _days_to_next(d, national_dates),
            "days_after_holiday": _days_since_last(d, national_dates),
        }

    prox_df = pd.DataFrame.from_dict(proximity_map, orient="index").reset_index()
    prox_df.columns = ["date", "days_to_holiday", "days_after_holiday"]

    df = df.merge(prox_df, on="date", how="left")

    # Cap at 30 days (beyond that, signal is noise)
    df["days_to_holiday"]    = df["days_to_holiday"].clip(upper=30)
    df["days_after_holiday"] = df["days_after_holiday"].clip(upper=30)

    return df


# ============================================================
# PROCESS TRANSACTIONS
# ============================================================

def process_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    transactions.csv has store-level daily footfall.
    Forward fill missing dates per store (closed store = no transactions).
    """
    # Some stores may have no transactions on some dates
    # Reindex to full date range per store
    min_date = transactions["date"].min()
    max_date = transactions["date"].max()
    full_range = pd.date_range(min_date, max_date, freq="D")

    filled = []
    for store_id, grp in transactions.groupby("store_nbr"):
        grp_indexed = grp.set_index("date").reindex(full_range)
        grp_indexed["store_nbr"] = store_id
        grp_indexed["transactions"] = grp_indexed["transactions"].ffill().bfill()
        filled.append(grp_indexed.reset_index().rename(columns={"index": "date"}))

    txn_full = pd.concat(filled, ignore_index=True)
    txn_full["store_nbr"] = txn_full["store_nbr"].astype(int)

    logger.info(
        "Transactions processed  |  shape=%s  |  missing=%d",
        txn_full.shape,
        txn_full["transactions"].isna().sum()
    )

    return txn_full


# ============================================================
# ADD EARTHQUAKE FLAG
# ============================================================

def add_earthquake_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ecuador earthquake: April 16, 2016.
    EDA showed ~+23% sales spike immediately after (panic buying).
    Flag the pre-event and post-event windows separately.
    """
    eq_date = pd.Timestamp(EARTHQUAKE_DATE)

    df["is_earthquake_pre"]  = (
        (df["date"] >= eq_date - pd.Timedelta(EARTHQUAKE_WINDOW_PRE, "D")) &
        (df["date"] <  eq_date)
    ).astype(int)

    df["is_earthquake_post"] = (
        (df["date"] >= eq_date) &
        (df["date"] <  eq_date + pd.Timedelta(EARTHQUAKE_WINDOW_POST, "D"))
    ).astype(int)

    return df


# ============================================================
# MERGE ALL — build full training DataFrame
# ============================================================

def build_merged_dataset(files: dict, is_train: bool = True) -> pd.DataFrame:
    """
    Merges train/test with stores, oil, holidays, transactions.

    Pipeline:
        1. Validate train/test schema
        2. Process oil (interpolate)
        3. Process holidays (flags)
        4. Process transactions (ffill per store)
        5. Merge stores metadata
        6. Add oil + holidays + transactions
        7. Add earthquake flag
        8. Add holiday proximity
        9. Encode store type

    Returns:
        Fully merged DataFrame ready for feature engineering.
    """
    base = files["train"] if is_train else files["test"]

    if is_train:
        base = validate_train(base)

    # ── Process supplementary files ───────────────────────────
    oil_clean  = process_oil(files["oil"])
    hol_clean  = process_holidays(files["holidays"])
    txn_clean  = process_transactions(files["transactions"])
    stores     = files["stores"].copy()

    # ── Encode store type → ordinal (A=4, B=3, C=2, D=1, E=0) ─
    type_map = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    stores["store_type_enc"] = stores["type"].map(type_map)
    stores = stores.rename(columns={"type": "store_type"})

    # ── Merge stores ──────────────────────────────────────────
    df = base.merge(stores, on="store_nbr", how="left")

    # ── Merge oil ─────────────────────────────────────────────
    df = df.merge(oil_clean[["date", "dcoilwtico"]], on="date", how="left")

    # ── Merge holidays ────────────────────────────────────────
    df = df.merge(hol_clean, on="date", how="left")

    # ── Fill holiday nulls → 0 (no holiday that day) ──────────
    hol_flag_cols = [
        "is_national_holiday", "is_regional_holiday", "is_local_holiday",
        "is_holiday_event", "is_transferred", "is_work_day",
        "is_any_holiday", "holiday_count"
    ]
    for col in hol_flag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # ── Merge transactions ────────────────────────────────────
    df = df.merge(
        txn_clean[["date", "store_nbr", "transactions"]],
        on=["date", "store_nbr"],
        how="left"
    )

    # ── Add earthquake flag ───────────────────────────────────
    df = add_earthquake_flag(df)

    # ── Add holiday proximity ─────────────────────────────────
    df = add_holiday_proximity(df, hol_clean)

    # ── Final sort ────────────────────────────────────────────
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    logger.info(
        "Merged dataset  |  shape=%s  |  columns=%d  |  date_range=%s → %s",
        df.shape,
        df.shape[1],
        df["date"].min().date(),
        df["date"].max().date()
    )

    return df


# ============================================================
# LOAD FULL PIPELINE — convenience function
# ============================================================

def load_and_merge(is_train: bool = True) -> pd.DataFrame:
    """
    Single entry point: load all files → merge → return DataFrame.
    Used by training_pipeline.py and forecast_api.py.
    """
    files = load_raw_files()
    return build_merged_dataset(files, is_train=is_train)
