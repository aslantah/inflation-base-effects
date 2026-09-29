"""Verify that optimised portfolios are exactly duration-neutral."""

import numpy as np
import pandas as pd

from inflation_base_effects.portfolio import optimize_portfolio


def test_duration_neutrality():
    """After optimisation, duration'w should be approximately zero."""
    n = 4
    cols = [f"A{i}" for i in range(n)]

    alpha = pd.Series([0.01, -0.005, 0.008, -0.003], index=cols)
    dur = pd.Series([8.0, 7.5, 8.2, 7.8], index=cols)

    rng = np.random.default_rng(42)
    A = rng.standard_normal((n, n)) * 0.01
    cov = pd.DataFrame(A @ A.T + np.eye(n) * 0.001, index=cols, columns=cols)

    w, status = optimize_portfolio(alpha, cov, dur, risk_aversion=1.0)
    dur_exposure = (dur * w).sum()
    assert abs(dur_exposure) < 1e-3, f"Duration exposure {dur_exposure:.6f} not neutral"
    assert status in ("optimal", "optimal_inaccurate"), f"Solver status: {status}"
