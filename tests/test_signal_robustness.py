"""Tests for the real-time month-of-year robustness signal."""

import numpy as np
import pandas as pd

from inflation_base_effects.signals import (
    calc_month_of_year_signal,
    expanding_month_of_year_zscore,
)


def _monthly_frame(periods: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", periods=periods, freq="MS")
    seasonal = np.tile(np.arange(1, 13, dtype=float), periods // 12 + 1)[:periods]
    trend = np.arange(periods, dtype=float) * 0.01
    return pd.DataFrame({"A": seasonal + trend}, index=dates)


def test_month_of_year_normalizer_uses_only_prior_same_month_values():
    values = _monthly_frame()
    baseline = expanding_month_of_year_zscore(values, min_history=5)
    altered = values.copy()
    altered.iloc[-1, 0] += 1_000
    recomputed = expanding_month_of_year_zscore(altered, min_history=5)
    pd.testing.assert_series_equal(baseline.iloc[:-1, 0], recomputed.iloc[:-1, 0])


def test_month_of_year_signal_is_lagged_and_capped():
    values = _monthly_frame()
    signal = calc_month_of_year_signal(values, values.index[-1], min_history=5, implag=1)
    assert signal.index.min() > values.index.min()
    assert signal.abs().max().max() <= 2.0
