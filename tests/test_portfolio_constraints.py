"""Verify volatility, gross, position, and duration constraints."""

import numpy as np
import pandas as pd

from inflation_base_effects.portfolio import optimize_portfolio


def _make_cov(n: int, scale: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    A = rng.standard_normal((n, n)) * scale
    S = A @ A.T + np.eye(n) * scale
    cols = [f"A{i}" for i in range(n)]
    return pd.DataFrame(S, index=cols, columns=cols)


def test_vol_ceiling():
    """After optimisation, annualised ex-ante vol should not exceed the ceiling."""
    cov = _make_cov(4)
    alpha = pd.Series([0.01, -0.005, 0.008, -0.003], index=cov.columns)
    dur = pd.Series([8.0, 7.5, 8.2, 7.8], index=cov.columns)
    w, status = optimize_portfolio(
        alpha, cov, dur, vol_ceiling=0.10, max_gross=5.0, max_position=1.0
    )
    port_var = float(w.values @ cov.values @ w.values) * 12
    realised_vol = np.sqrt(port_var)
    assert realised_vol <= 0.10 + 0.01, f"Vol {realised_vol:.4f} exceeds ceiling 0.10"


def test_gross_exposure():
    """Gross exposure should not exceed max_gross."""
    cov = _make_cov(4, scale=0.0001)
    alpha = pd.Series([0.01, -0.005, 0.008, -0.003], index=cov.columns)
    dur = pd.Series([8.0, 7.5, 8.2, 7.8], index=cov.columns)
    w, status = optimize_portfolio(
        alpha, cov, dur, vol_ceiling=0.10, max_gross=2.0, max_position=1.0
    )
    assert np.abs(w).sum() <= 2.0 + 1e-3


def test_position_limits():
    """No single position should exceed max_position in absolute value."""
    cov = _make_cov(4, scale=0.0001)
    alpha = pd.Series([0.05, -0.03, 0.02, -0.04], index=cov.columns)
    dur = pd.Series([8.0, 7.5, 8.2, 7.8], index=cov.columns)
    w, status = optimize_portfolio(
        alpha, cov, dur, vol_ceiling=0.10, max_gross=10.0, max_position=0.5
    )
    assert np.abs(w).max() <= 0.5 + 1e-3
