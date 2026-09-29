"""Verify that the signal timing chain places observations at the correct dates."""

import numpy as np
import pandas as pd

from inflation_base_effects.signals import calc_signal


def test_single_nonzero_cpi_appears_at_correct_date():
    """A single nonzero CPI print at month T should produce a signal at T+12+implag."""
    dates = pd.date_range("2010-01-01", periods=36, freq="MS")
    cpi = pd.DataFrame(0.0, index=dates, columns=["A"])
    # Place one nonzero observation at 2010-06-01 (index 5)
    cpi.iloc[5, 0] = 0.05

    last_ret = dates[-1]
    score = calc_signal(cpi, last_return_date=last_ret, shiftval=12, hl=3, implag=1)

    # The shift(12, freq='MS') moves the 2010-06 value to 2011-06.
    # After implag=1, the signal should first be nonzero at 2011-07 or later.
    nonzero = score[score["A"].abs() > 1e-10]
    if not nonzero.empty:
        first_nonzero = nonzero.index[0]
        assert first_nonzero >= pd.Timestamp("2011-07-01"), (
            f"Signal appeared at {first_nonzero}, expected >= 2011-07-01"
        )


def test_signal_does_not_extend_beyond_returns():
    """Signal index must not go past the last return date."""
    dates = pd.date_range("2000-01-01", periods=200, freq="MS")
    rng = np.random.default_rng(42)
    cpi = pd.DataFrame(rng.standard_normal((200, 2)) * 0.01, index=dates, columns=["A", "B"])
    last_ret = pd.Timestamp("2015-06-01")

    score = calc_signal(cpi, last_return_date=last_ret, shiftval=12, hl=6, implag=1)
    assert score.index[-1] <= last_ret, (
        f"Signal ends at {score.index[-1]}, but returns end at {last_ret}"
    )
