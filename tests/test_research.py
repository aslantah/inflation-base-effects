"""Economic identities, aligned research diagnostics, and reduced universes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inflation_base_effects.evaluation import portfolio_returns
from inflation_base_effects.portfolio import compute_bond_returns, mod_duration
from inflation_base_effects.research import (
    benchmark_regression,
    comparison_table,
    constraint_summary,
    leave_one_out,
    mechanical_example,
    mechanism_regression,
    performance_summary,
    portfolio_attribution,
    realized_turnover,
    return_components,
    run_experiment,
    sensitivity_experiments,
)
from inflation_base_effects.signals import calc_signal_from_base_effect


def test_mechanical_departure_and_offset():
    example = mechanical_example()
    np.testing.assert_allclose(example.loc[-12:, "price"], 101)
    np.testing.assert_allclose(example.loc[-12:-1, "yoy_pct"], 1)
    np.testing.assert_allclose(example.loc[0:, "yoy_pct"], 0)
    assert example.loc[-12, "change_yoy_pp"] == pytest.approx(1)
    assert example.loc[0, "change_yoy_pp"] == pytest.approx(-1)
    np.testing.assert_allclose(example.drop(index=[-12, 0]).change_yoy_pp, 0)
    offset = mechanical_example(True)
    assert offset.loc[0, "price"] == pytest.approx(102.01)
    assert offset.loc[0, "yoy_pct"] == pytest.approx(1)
    assert offset.loc[0, "change_yoy_pp"] == pytest.approx(0)
    assert offset.loc[12, "change_yoy_pp"] == pytest.approx(-1)


def test_duration_continuity_and_domain():
    for yield_pct in [-2, -0.5, -0.000001, 0, 0.000001, 0.1, 5]:
        result = mod_duration(yield_pct)
        assert np.isfinite(result) and result > 0
        assert abs(result - mod_duration(yield_pct + 1e-8)) < 1e-6
    assert mod_duration(0) == 10
    assert mod_duration(-0.5) > 10
    assert np.isnan(mod_duration(np.nan))
    for value in [-200, -201, np.inf]:
        with pytest.raises(ValueError):
            mod_duration(value)
    # Independent finite cash-flow valuation check at positive par yields.
    for yield_pct in [0.05, 0.1, 3]:
        y = yield_pct / 100
        periods = np.arange(1, 21)
        flows = np.full(20, y / 2)
        flows[-1] += 1
        pv = flows / (1 + y / 2) ** periods
        expected = ((periods / 2) * pv).sum() / pv.sum() / (1 + y / 2)
        assert mod_duration(yield_pct) == pytest.approx(expected)


def test_implementation_delay_preserves_final_available_print():
    dates = pd.date_range("2000-01-01", periods=12, freq="MS")
    old_prints = pd.DataFrame({"US": np.arange(12, dtype=float)}, index=dates)
    score = calc_signal_from_base_effect(old_prints, pd.Timestamp("2001-01-01"))
    assert score.index[-1] == pd.Timestamp("2001-01-01")
    truncated = calc_signal_from_base_effect(old_prints, dates[-1])
    assert truncated.index[-1] == dates[-1]


@pytest.fixture
def market():
    rng = np.random.default_rng(527)
    dates = pd.date_range("2000-01-01", periods=60, freq="MS")
    cols = ["US", "DE", "UK", "CA"]
    yields = pd.DataFrame(
        2 + rng.normal(0, 0.08, (60, 4)).cumsum(axis=0), index=dates, columns=cols
    )
    cpi = pd.DataFrame(
        100 * np.exp(rng.normal(0.002, 0.003, (60, 4)).cumsum(axis=0)), index=dates, columns=cols
    )
    return cpi, yields


def test_attribution_and_missing_data(market):
    _, yields = market
    holdings = yields * 0 + [0.3, -0.3, 0.2, -0.2]
    yields.iloc[20, 0] = np.nan
    returns, duration = compute_bond_returns(yields)
    carry, price = return_components(yields, duration)
    pd.testing.assert_frame_equal((carry + price).dropna(how="all"), returns)
    country, components = portfolio_attribution(holdings, yields)
    expected = portfolio_returns(holdings, returns)
    np.testing.assert_allclose(country.sum(axis=1), expected)
    np.testing.assert_allclose(components.carry + components.yield_change, expected)
    np.testing.assert_allclose(components.tilt + components.timing, expected)
    assert yields.index[20] not in country.index
    assert yields.index[21] not in country.index


def test_turnover_cost_timing_and_common_dates():
    dates = pd.date_range("2000-01-01", periods=6, freq="MS")
    holdings = pd.DataFrame({"US": [1, 2, 2, 0, 1, 1]}, index=dates)
    turnover = realized_turnover(holdings)
    np.testing.assert_allclose(turnover.iloc[1:], [1, 1, 0, 2, 1])
    gross = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01], index=dates[1:])
    net = gross - turnover.reindex(gross.index) * 5 / 10000
    assert net.iloc[0] == pytest.approx(0.0095)
    table = comparison_table({"A": gross, "B": gross.iloc[1:]})
    assert table.N.tolist() == [4, 4]
    full = comparison_table({"A": gross, "B": gross.iloc[1:]}, common=False)
    assert full.N.tolist() == [5, 4]
    stats = performance_summary(pd.Series([-0.1, 0.02, -0.01], index=dates[:3]))
    assert stats.max_additive_drawdown == pytest.approx(-0.1)


def test_benchmark_regression_known_relation():
    dates = pd.date_range("2000-01-01", periods=60, freq="MS")
    rng = np.random.default_rng(25)
    benchmark = pd.Series(rng.normal(0, 0.02, 60), index=dates)
    result = benchmark_regression(0.001 + 0.4 * benchmark, benchmark.iloc[2:])
    assert result.N == 58
    assert result.beta == pytest.approx(0.4)
    assert result.ann_intercept == pytest.approx(0.012)


def test_mechanism_preserves_distinct_surveys_in_same_deadline_month():
    dates = pd.date_range("2000-01-01", periods=5, freq="MS")
    base = pd.Series([1, 2, 4, 3, 5], index=dates)
    revision = pd.Series([1.1, 1.2, 1.8, 3, 2, 4], index=dates.insert(1, dates[0]))
    result = mechanism_regression(base, revision)
    assert result.N == 6
    assert np.isfinite(result.beta)


def test_leave_one_out_reoptimizes_and_preserves_constraints(market):
    cpi, yields = market
    variants = leave_one_out(cpi, yields)
    assert len(variants) == 4
    for name, experiment in variants.items():
        assert name.removeprefix("Without ") not in experiment.holdings.columns
        assert len(experiment.holdings.columns) == 3
        summary = constraint_summary(experiment.diagnostics)
        assert summary.failed == 0
        assert summary.max_duration_residual < 1e-3
        assert summary.max_position <= 0.501
        assert summary.max_gross <= 2.001
        assert summary.max_forecast_vol <= 0.101


def test_research_backtest_has_no_future_dependency(market):
    cpi, yields = market
    full = run_experiment(cpi, yields)
    end = cpi.index[45]
    truncated = run_experiment(cpi.loc[:end], yields.loc[:end])
    pd.testing.assert_frame_equal(full.holdings.loc[:end], truncated.holdings)


def test_sensitivity_grid_is_one_at_a_time(monkeypatch, market):
    calls = []
    sentinel = object()

    def fake(*args, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr("inflation_base_effects.research.run_experiment", fake)
    groups = sensitivity_experiments(*market, baseline=sentinel)
    assert [len(group) for group in groups.values()] == [3, 4, 3, 3]
    assert len(calls) == 9
    assert all(len(kwargs) == 1 for kwargs in calls)


def test_supplementary_snapshot_checksums():
    root = Path("data/supplementary/2026-10-01")
    manifest = json.loads((root / "manifest.json").read_text())
    assert set(manifest["sha256"]) == {"GS10.csv", "DGS10.csv"}
    for name, checksum in manifest["sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == checksum
