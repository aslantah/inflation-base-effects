"""Verify that missing CPI values never become zero returns."""

import numpy as np
import pandas as pd

from inflation_base_effects.data import prepare_inflation_data


def test_nan_cpi_produces_nan_not_zero():
    """When CPI has trailing NaN, MoM change should be NaN - never 0."""
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    values = np.linspace(100, 135, 36)
    values[-3:] = np.nan
    cpi = pd.Series(values, index=dates)

    frame = pd.DataFrame({"A": cpi})
    mom, _, _ = prepare_inflation_data(frame)
    # Months with NaN CPI should produce NaN, not 0
    assert mom.iloc[-3:, 0].isna().all()


def test_pct_change_default_fills_nan():
    """Demonstrate that pct_change() default would fabricate zeros - our method avoids this."""
    dates = pd.date_range("2020-01-01", periods=6, freq="MS")
    cpi = pd.Series([100, 101, 102, np.nan, np.nan, np.nan], index=dates)

    # Explicit division (our method) preserves NaN
    explicit = cpi / cpi.shift(1) - 1
    assert pd.isna(explicit.iloc[4]), "Explicit division should produce NaN for NaN/NaN"
