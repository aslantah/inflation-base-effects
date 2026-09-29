"""Inference and no-lookahead tests for the temporal-stability study."""

import numpy as np
import pandas as pd
import pytest

from inflation_base_effects.evaluation import (
    hac_mean_inference,
    regime_stability,
    rolling_hac_mean,
)


def test_hac_mean_inference_annualizes_and_drops_missing_values():
    dates = pd.date_range("2000-01-01", periods=25, freq="MS")
    returns = pd.Series(np.linspace(-0.01, 0.02, 25), index=dates)
    returns.iloc[[3, 9]] = np.nan

    result = hac_mean_inference(returns)

    clean = returns.dropna()
    assert result["N"] == 23
    assert result["ann_return"] == pytest.approx(12 * clean.mean())
    assert result["ann_vol"] == pytest.approx(np.sqrt(12) * clean.std())
    assert result["CI 95% lower"] < result["ann_return"] < result["CI 95% upper"]


def test_regime_stability_recovers_post_minus_pre_direction():
    dates = pd.date_range("2000-01-01", periods=240, freq="MS")
    variation = 0.002 * np.sin(np.arange(240))
    returns = pd.Series(0.001 + variation, index=dates)
    returns.iloc[120:] += 0.004

    result = regime_stability(returns, "2010-01-01")

    assert list(result.index) == ["pre", "post", "post_minus_pre"]
    assert result.loc["pre", "N"] == 120
    assert result.loc["post", "N"] == 120
    assert result.loc["post_minus_pre", "ann_return"] == pytest.approx(0.048, abs=5e-4)
    assert result.loc["post_minus_pre", "NW t-stat"] > 0
    assert result.loc["post_minus_pre", "NW p-value"] < 0.05


def test_regime_stability_rejects_undersized_segments():
    dates = pd.date_range("2000-01-01", periods=30, freq="MS")
    returns = pd.Series(np.linspace(-0.01, 0.01, 30), index=dates)

    with pytest.raises(ValueError, match="at least twelve"):
        regime_stability(returns, "2000-06-01")


def test_rolling_hac_mean_has_no_future_dependence():
    dates = pd.date_range("2000-01-01", periods=180, freq="MS")
    returns = pd.Series(0.001 + 0.01 * np.sin(np.arange(180)), index=dates)
    baseline = rolling_hac_mean(returns, window=120)

    perturbed = returns.copy()
    perturbed.iloc[150:] += 1.0
    changed = rolling_hac_mean(perturbed, window=120)

    pd.testing.assert_frame_equal(baseline.loc[: dates[149]], changed.loc[: dates[149]])
    assert not baseline.loc[dates[150] :].equals(changed.loc[dates[150] :])


def test_temporal_helpers_validate_inputs():
    dates = pd.date_range("2000-01-01", periods=10, freq="MS")
    returns = pd.Series(np.arange(10, dtype=float), index=dates)

    with pytest.raises(ValueError, match="at least three"):
        hac_mean_inference(returns.iloc[:2])
    with pytest.raises(ValueError, match="shorter"):
        rolling_hac_mean(returns, window=12)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        hac_mean_inference(pd.Series([0.1, 0.2, 0.3]))
