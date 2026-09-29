"""Tests for the optimizer-free rank/DV01 benchmark."""

import numpy as np
import pandas as pd

from inflation_base_effects.portfolio import rank_dv01_holdings


def test_rank_benchmark_direction_gross_and_dv01_neutrality():
    date = pd.Timestamp("2024-01-01")
    columns = ["A", "B", "C", "D"]
    score = pd.DataFrame([[4.0, 3.0, 2.0, 1.0]], index=[date], columns=columns)
    duration = pd.DataFrame([[8.0, 7.0, 9.0, 6.0]], index=[date], columns=columns)
    holdings = rank_dv01_holdings(score, duration)
    row = holdings.loc[date]
    assert row["A"] > 0 and row["B"] > 0
    assert row["C"] < 0 and row["D"] < 0
    assert np.isclose(row.abs().sum(), 1.0)
    assert np.isclose((row * duration.loc[date]).sum(), 0.0, atol=1e-12)
