"""Assert that no signal values are generated beyond the last observable return."""

import numpy as np
import pandas as pd

from inflation_base_effects.signals import calc_signal


def test_no_lookahead():
    """calc_signal must truncate at last_return_date."""
    dates = pd.date_range("2000-01-01", periods=300, freq="MS")
    rng = np.random.default_rng(42)
    cpi = pd.DataFrame(rng.standard_normal((300, 3)) * 0.02, index=dates, columns=["A", "B", "C"])
    last_ret = pd.Timestamp("2020-01-01")

    score = calc_signal(cpi, last_return_date=last_ret, shiftval=12, hl=6, implag=1)

    assert score.index[-1] <= last_ret
    # Also: no NaN-only rows should survive
    assert not score.dropna(how="all").empty
