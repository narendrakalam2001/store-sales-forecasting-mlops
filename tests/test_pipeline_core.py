# ============================================================
# PYTEST UNIT TESTS — Store Sales Forecasting System
# ============================================================
# Run with:  pytest tests/test_pipeline_core.py -v
#
# Covers:
#   - Clipper transformer
#   - SmartImputer
#   - classify_feature_columns
#   - drop_lag_nans
#   - walk_forward_splits
#   - RMSLE / RMSE / MAE / MAPE
#   - PSI (same fixed version as Credit Risk)
#   - get_confidence_band
#   - apply_business_rules
#   - score_forecast
#   - add_calendar_features
#   - add_lag_features
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
import pandas as pd

from src.preprocessing      import Clipper, SmartImputer, classify_feature_columns, drop_lag_nans, walk_forward_splits
from src.metrics             import rmsle, rmse, mae, mape, psi
from src.forecast_engine     import get_confidence_band, apply_business_rules, score_forecast
from src.feature_engineering import add_calendar_features, add_lag_features


# ============================================================
# CLIPPER TESTS
# ============================================================

class TestClipper:

    def test_output_shape_matches_input(self):
        X = np.array([[1.0], [2.0], [3.0], [9999.0]])
        c = Clipper(fold=1.5).fit(X)
        assert c.transform(X).shape == X.shape

    def test_clips_extreme_outlier(self):
        X = np.array([[1.0], [2.0], [3.0], [9999.0]])
        c = Clipper(fold=1.5).fit(X)
        assert c.transform(X).max() < 9999.0

    def test_no_change_on_normal_data(self):
        X = np.array([[10.0], [11.0], [12.0], [13.0]])
        c = Clipper(fold=1.5).fit(X)
        np.testing.assert_array_almost_equal(X, c.transform(X), decimal=3)

    def test_feature_names_out(self):
        X = np.array([[1.0], [2.0]])
        c = Clipper().fit(X)
        names = c.get_feature_names_out(["income"])
        assert list(names) == ["income"]


# ============================================================
# SMART IMPUTER TESTS
# ============================================================

class TestSmartImputer:

    def test_fills_nan_with_median_when_outliers(self):
        # Column with extreme outlier → should use MEDIAN
        X = np.array([[1.0], [2.0], [3.0], [9999.0], [np.nan]])
        imp = SmartImputer().fit(X[:-1])
        result = imp.transform(X)
        # NaN should be filled, not remain NaN
        assert not np.isnan(result).any()

    def test_fills_nan_with_mean_when_no_outliers(self):
        X = np.array([[10.0], [11.0], [12.0], [13.0], [np.nan]])
        imp = SmartImputer().fit(X[:-1])
        result = imp.transform(X)
        assert not np.isnan(result).any()
        # Mean of [10,11,12,13] = 11.5
        assert abs(result[-1, 0] - 11.5) < 0.01

    def test_no_change_on_clean_data(self):
        X = np.array([[1.0], [2.0], [3.0]])
        imp = SmartImputer().fit(X)
        np.testing.assert_array_equal(X, imp.transform(X))


# ============================================================
# CLASSIFY FEATURE COLUMNS TESTS
# ============================================================

class TestClassifyFeatureColumns:

    def _sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "is_weekend":      [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            "dow_sin":         np.sin(np.linspace(0, 2*np.pi, 10)),
            "sales_lag_7":     np.random.exponential(scale=50, size=10),
            "year":            [2015]*5 + [2016]*5,
            "rolling_mean_7d": np.random.normal(100, 10, 10),
        })

    def test_binary_detected(self):
        df     = self._sample_df()
        groups = classify_feature_columns(df, list(df.columns))
        assert "is_weekend" in groups["binary"]

    def test_cyclical_detected(self):
        df     = self._sample_df()
        groups = classify_feature_columns(df, list(df.columns))
        assert "dow_sin" in groups["cyclical"]

    def test_returns_all_groups(self):
        df     = self._sample_df()
        groups = classify_feature_columns(df, list(df.columns))
        for key in ["binary", "cyclical", "skewed", "normal", "ordinal"]:
            assert key in groups


# ============================================================
# DROP LAG NANS TESTS
# ============================================================

class TestDropLagNans:

    def test_drops_nan_rows(self):
        df = pd.DataFrame({
            "sales_lag_28": [np.nan, np.nan, 10.0, 20.0, 30.0]
        })
        result = drop_lag_nans(df, lag_col="sales_lag_28")
        assert len(result) == 3
        assert not result["sales_lag_28"].isna().any()

    def test_no_drop_when_no_nans(self):
        df = pd.DataFrame({"sales_lag_28": [1.0, 2.0, 3.0]})
        result = drop_lag_nans(df, lag_col="sales_lag_28")
        assert len(result) == 3

    def test_missing_col_returns_unchanged(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = drop_lag_nans(df, lag_col="sales_lag_28")
        assert len(result) == 3


# ============================================================
# WALK-FORWARD SPLITS TESTS
# ============================================================

class TestWalkForwardSplits:

    def _sample_df(self):
        dates = pd.date_range("2016-01-01", periods=100, freq="D")
        return pd.DataFrame({
            "date":  dates,
            "sales": np.random.rand(100),
        })

    def test_returns_correct_number_of_splits(self):
        df     = self._sample_df()
        splits = walk_forward_splits(df, n_splits=3, val_days=10)
        assert len(splits) == 3

    def test_train_before_val(self):
        df     = self._sample_df()
        splits = walk_forward_splits(df, n_splits=2, val_days=10)
        for train_mask, val_mask, val_start, val_end in splits:
            train_dates = df[train_mask]["date"]
            val_dates   = df[val_mask]["date"]
            assert train_dates.max() < val_dates.min()

    def test_no_overlap_between_train_and_val(self):
        df     = self._sample_df()
        splits = walk_forward_splits(df, n_splits=2, val_days=10)
        for train_mask, val_mask, _, _ in splits:
            assert (train_mask & val_mask).sum() == 0


# ============================================================
# METRICS TESTS
# ============================================================

class TestRMSLE:

    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0])
        assert rmsle(y, y) < 1e-9

    def test_zero_sales_handled(self):
        y_true = np.array([0.0, 10.0, 20.0])
        y_pred = np.array([0.0, 10.0, 20.0])
        assert rmsle(y_true, y_pred) < 1e-9

    def test_higher_error_gives_higher_rmsle(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred_good = np.array([11.0, 21.0, 31.0])
        y_pred_bad  = np.array([20.0, 40.0, 60.0])
        assert rmsle(y_true, y_pred_good) < rmsle(y_true, y_pred_bad)

    def test_negative_pred_clipped(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([-5.0, 20.0])
        score  = rmsle(y_true, y_pred)
        assert score > 0

    def test_returns_float(self):
        assert isinstance(rmsle(np.array([1.0]), np.array([1.0])), float)


class TestOtherMetrics:

    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) < 1e-9

    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) < 1e-9

    def test_mape_skips_zeros(self):
        y_true = np.array([0.0, 0.0, 10.0])
        y_pred = np.array([5.0, 5.0, 10.0])
        score  = mape(y_true, y_pred, eps=1.0)
        assert score < 1e-9

    def test_psi_identical_distributions(self):
        x = np.random.normal(0, 1, 500)
        assert psi(x, x) < 0.05

    def test_psi_shifted_higher(self):
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, 1000)
        new = rng.normal(3, 1, 1000)
        assert psi(ref, new) > psi(ref, ref)


# ============================================================
# FORECAST ENGINE TESTS
# ============================================================

class TestConfidenceBand:

    def test_high_confidence_no_uncertainty(self):
        band = get_confidence_band(pred=100.0, family_zero_pct=0.1,
                                   is_holiday=0, is_promoted=0,
                                   days_since_train_end=0)
        assert band == "HIGH"

    def test_low_confidence_many_factors(self):
        band = get_confidence_band(pred=10.0, family_zero_pct=0.9,
                                   is_holiday=1, is_promoted=1,
                                   days_since_train_end=60)
        assert band == "LOW"

    def test_medium_confidence_some_factors(self):
        band = get_confidence_band(pred=50.0, family_zero_pct=0.6,
                                   is_holiday=0, is_promoted=0,
                                   days_since_train_end=0)
        assert band == "MEDIUM"


class TestBusinessRules:

    def test_store_closed_overrides_all(self):
        pred, rule = apply_business_rules(pred=100.0, is_store_closed=1)
        assert pred == 0.0
        assert rule == "STORE_CLOSED"

    def test_high_zero_family_no_promo_floors(self):
        pred, rule = apply_business_rules(
            pred=50.0, family_zero_pct=0.95, is_promoted=0, is_store_closed=0
        )
        assert pred == 0.0
        assert rule == "HIGH_ZERO_FAMILY_NO_PROMO"

    def test_negative_prediction_clipped(self):
        pred, rule = apply_business_rules(pred=-10.0, is_store_closed=0)
        assert pred == 0.0
        assert rule == "NEGATIVE_CLIP"

    def test_normal_prediction_unchanged(self):
        pred, rule = apply_business_rules(
            pred=100.0, family_zero_pct=0.1, is_promoted=0, is_store_closed=0
        )
        assert pred == 100.0
        assert rule is None


class TestScoreForecast:

    def test_output_keys_complete(self):
        result = score_forecast(
            pred=50.0, store_nbr=1, family="GROCERY I",
            forecast_date  = pd.Timestamp("2017-08-16"),
            train_end_date = pd.Timestamp("2017-08-15"),
            family_zero_pct=0.1, is_holiday=0, is_promoted=0,
        )
        for key in ["store_nbr","family","forecast_date","predicted_sales",
                    "confidence_band","rule_triggered","is_holiday_forecast",
                    "is_promo_forecast","days_ahead"]:
            assert key in result

    def test_days_ahead_computed(self):
        result = score_forecast(
            pred=50.0, store_nbr=1, family="BEVERAGES",
            forecast_date  = pd.Timestamp("2017-08-20"),
            train_end_date = pd.Timestamp("2017-08-15"),
        )
        assert result["days_ahead"] == 5


# ============================================================
# FEATURE ENGINEERING TESTS
# ============================================================

class TestCalendarFeatures:

    def _sample(self):
        return pd.DataFrame({
            "date":  pd.date_range("2017-01-01", periods=10, freq="D"),
            "sales": np.random.rand(10),
        })

    def test_calendar_columns_added(self):
        df     = self._sample()
        result = add_calendar_features(df)
        for col in ["day_of_week","month","year","is_weekend","week_of_year"]:
            assert col in result.columns

    def test_is_weekend_correct(self):
        df     = self._sample()
        result = add_calendar_features(df)
        for _, row in result.iterrows():
            expected = int(row["date"].dayofweek >= 5)
            assert row["is_weekend"] == expected


class TestLagFeatures:

    def _sample(self):
        stores   = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        families = ["GROCERY I"] * 10
        return pd.DataFrame({
            "store_nbr": stores,
            "family":    families,
            "date":      pd.date_range("2017-01-01", periods=10, freq="D").tolist() * 1,
            "sales":     np.arange(10, dtype=float),
        })

    def test_lag_columns_created(self):
        df     = self._sample()
        result = add_lag_features(df, lag_days=[1, 7])
        assert "sales_lag_1" in result.columns
        assert "sales_lag_7" in result.columns

    def test_lag_1_correct_value(self):
        df     = self._sample()
        result = add_lag_features(df, lag_days=[1])
        # Within store 1 group, lag_1 of row index 1 = sales[0]
        store1 = result[result["store_nbr"] == 1].sort_values("date").reset_index(drop=True)
        assert np.isnan(store1.loc[0, "sales_lag_1"])   # first row = NaN
        assert store1.loc[1, "sales_lag_1"] == store1.loc[0, "sales"]
