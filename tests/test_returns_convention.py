"""Verify that returns use prior-period holdings (h.shift(1) * ret)."""

import numpy as np
import pandas as pd

from inflation_base_effects.evaluation import EmpPerfStats, portfolio_returns


def test_holdings_shifted_before_multiplying_returns():
    """Portfolio return at time t should use holdings from time t-1."""
    dates = pd.date_range("2020-01-01", periods=5, freq="MS")
    h = pd.DataFrame({"A": [0, 1, 0, 1, 0]}, index=dates, dtype=float)
    ret = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=dates, dtype=float)

    perf = EmpPerfStats(h, ret)
    # The first date has no prior holding and must be excluded, not recorded as zero.
    expected = np.array([0.0, 0.03, 0.0, 0.05])
    np.testing.assert_array_almost_equal(perf.port_ret.values, expected)
    assert perf.port_ret.index[0] == dates[1]


def test_partial_return_row_is_excluded():
    dates = pd.date_range("2020-01-01", periods=4, freq="MS")
    holdings = pd.DataFrame({"A": [1.0] * 4, "B": [-1.0] * 4}, index=dates)
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, 0.03, 0.04], "B": [0.01, np.nan, 0.01, 0.01]},
        index=dates,
    )
    result = portfolio_returns(holdings, returns)
    assert dates[1] not in result.index
    assert list(result.index) == [dates[2], dates[3]]
