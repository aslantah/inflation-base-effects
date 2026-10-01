"""Portfolio construction: covariance, alpha, optimisation, backtest."""

from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
import pandas as pd

from .signals import univ_align

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bond return approximation
# ---------------------------------------------------------------------------


def mod_duration(y_pct: float, maturity: int = 10) -> float:
    """Modified Duration for a par bond with semiannual coupons.

    Uses the closed-form: D_mod = (1/y) [1 - (1 + y/2)^(-2T)]
    """
    y = y_pct / 100
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if pd.isna(y):
        return float("nan")
    if not np.isfinite(y) or y <= -2:
        raise ValueError("yield must be finite and greater than -200 percent")
    if y == 0:
        return float(maturity)
    # Stable at zero and continuous for negative yields; the zero limit is T.
    # A negative-coupon par bond is an algebraic proxy, not a tradable instrument.
    return float(-np.expm1(-2 * maturity * np.log1p(y / 2)) / y)


def compute_bond_returns(yields_monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Duration-adjusted bond return approximation from government bond yields.

    Returns (returns, duration) DataFrames.

    Monthly total return = carry + price return
      carry = yield / 12        (coupon income for a par bond)
      price = -D_mod * Delta y  (first-order duration approximation)

    This is the standard academic approach (Ilmanen 2011) and avoids the need
    for proprietary futures or total-return index data.
    """
    try:
        dur = yields_monthly.map(mod_duration)
    except AttributeError:  # pandas < 2.1
        dur = yields_monthly.applymap(mod_duration)  # type: ignore[attr-defined]

    dy = yields_monthly.diff() / 100  # yield change in decimal
    carry = yields_monthly.shift(1) / 100 / 12  # monthly carry
    ret = (carry - dur.shift(1) * dy).dropna(how="all")
    return ret, dur


def rank_dv01_holdings(
    score: pd.DataFrame,
    duration: pd.DataFrame,
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """Build an optimizer-free, rank-based, DV01-neutral portfolio.

    For an even common universe, the upper half is long and the lower half is
    short.  Inverse-duration notionals give each selected asset equal absolute
    DV01.  Equal numbers on each side make total DV01 zero, and a common scale
    sets the requested gross exposure.
    """
    if gross_exposure <= 0:
        raise ValueError("gross_exposure must be positive")
    score, duration = univ_align(score, duration)
    if len(score.columns) < 4 or len(score.columns) % 2:
        raise ValueError("rank_dv01_holdings requires an even universe of at least four assets")

    holdings = pd.DataFrame(np.nan, index=score.index, columns=score.columns, dtype=float)
    half = len(score.columns) // 2
    for date in score.index:
        signal = score.loc[date]
        dur = duration.loc[date]
        if signal.isna().any() or dur.isna().any():
            continue
        if (dur <= 0).any():
            raise ValueError(f"Durations must be positive at {date:%Y-%m-%d}")
        ordered = signal.sort_values(kind="stable")
        short_assets = ordered.index[:half]
        long_assets = ordered.index[-half:]
        inverse_duration = 1.0 / dur
        scale = gross_exposure / inverse_duration.sum()
        row = pd.Series(0.0, index=score.columns)
        row.loc[long_assets] = scale * inverse_duration.loc[long_assets]
        row.loc[short_assets] = -scale * inverse_duration.loc[short_assets]
        holdings.loc[date] = row
    return holdings.dropna(how="all")


# ---------------------------------------------------------------------------
# EWMA covariance with shrinkage
# ---------------------------------------------------------------------------


def ewmacov(
    ret: pd.DataFrame,
    halflife: tuple[int, int] = (12, 36),
    shrinkage_factor: float = 0.3,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Blended EWMA covariance: (1-s) short + s long.

    Returns ``{date: covariance_DataFrame}``.
    """
    hl_s, hl_l = halflife
    a_s = 1 - np.exp(-np.log(2) / hl_s)
    a_l = 1 - np.exp(-np.log(2) / hl_l)
    assets = ret.columns.tolist()
    n = len(assets)
    cov_s = cov_l = np.zeros((n, n))
    mu_s = mu_l = np.zeros(n)
    result: dict[pd.Timestamp, pd.DataFrame] = {}
    count = 0

    for date, row in ret.iterrows():
        r = row.values.astype(float)
        if np.any(np.isnan(r)):
            continue
        mu_s = (1 - a_s) * mu_s + a_s * r
        mu_l = (1 - a_l) * mu_l + a_l * r
        cov_s = (1 - a_s) * cov_s + a_s * np.outer(r - mu_s, r - mu_s)
        cov_l = (1 - a_l) * cov_l + a_l * np.outer(r - mu_l, r - mu_l)
        count += 1
        if count >= max(hl_s, 6):
            result[date] = pd.DataFrame(
                (1 - shrinkage_factor) * cov_s + shrinkage_factor * cov_l,
                index=assets,
                columns=assets,
            )
    return result


# ---------------------------------------------------------------------------
# Alpha from signal (Grinold-Kahn fundamental law)
# ---------------------------------------------------------------------------


def compute_alpha(
    score: pd.DataFrame,
    cov_dict: dict[pd.Timestamp, pd.DataFrame],
    ic: float = 0.05,
) -> pd.DataFrame:
    """alpha_i = IC * sigma_i * z_i"""
    alphas = pd.DataFrame(index=score.index, columns=score.columns, dtype=float)
    dates = sorted(cov_dict.keys())
    for dt in score.index:
        valid = [d for d in dates if d <= dt]
        if not valid:
            continue
        cov = cov_dict[valid[-1]]
        cols = score.columns.intersection(cov.columns)
        if len(cols) == 0:
            continue
        vols = np.sqrt(np.diag(cov.loc[cols, cols].values))
        alphas.loc[dt, cols] = ic * vols * score.loc[dt, cols].values
    return alphas.dropna(how="all").astype(float)


# ---------------------------------------------------------------------------
# Mean-variance optimisation
# ---------------------------------------------------------------------------


def optimize_portfolio(
    alpha_t: pd.Series,
    cov_t: pd.DataFrame,
    dur_t: pd.Series,
    risk_aversion: float = 1.0,
    vol_ceiling: float = 0.10,
    max_gross: float = 2.0,
    max_position: float = 0.5,
    ann_factor: int = 12,
) -> tuple[pd.Series, str]:
    """Constrained mean-variance optimisation.

    max alpha'w - (lambda/2) w'Sigma w
    s.t. duration'w = 0              (exact duration neutrality)
         ||w||_1 <= max_gross        (gross exposure ceiling)
         |w_i| <= max_pos            (position limits)
         w'Sigma w <= sigma_ceiling^2 / ann_factor  (annualised vol ceiling)

    All constraints are enforced *inside* the optimiser so that the solution
    is jointly feasible - no post-optimisation clipping is needed.

    Returns (weights_series, solver_status_string).
    """
    n = len(alpha_t)
    w = cp.Variable(n)
    S = (cov_t.values.astype(float) + cov_t.values.astype(float).T) / 2 + np.eye(n) * 1e-8
    obj = cp.Maximize(alpha_t.values.astype(float) @ w - (risk_aversion / 2) * cp.quad_form(w, S))
    constraints = [
        dur_t.values.astype(float) @ w == 0,  # duration neutrality
        cp.norm(w, 1) <= max_gross,  # gross exposure ceiling
        w >= -max_position,  # position floor
        w <= max_position,  # position ceiling
        cp.quad_form(w, S) <= (vol_ceiling**2) / ann_factor,  # vol ceiling
    ]
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return pd.Series(w.value, index=alpha_t.index), prob.status
    except cp.error.SolverError as exc:
        logger.warning("Solver failed at %s: %s", alpha_t.name, exc)
    return pd.Series(0.0, index=alpha_t.index), "failed"


def run_backtest(
    alpha: pd.DataFrame,
    cov_dict: dict[pd.Timestamp, pd.DataFrame],
    ret: pd.DataFrame,
    duration: pd.DataFrame,
    risk_aversion: float = 1.0,
    vol_ceiling: float = 0.10,
    max_gross: float = 2.0,
    max_position: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling mean-variance optimisation with all constraints inside the solver.

    Returns (holdings_DataFrame, diagnostics_DataFrame).
    The diagnostics table contains per-date constraint residuals and solver
    status, computed from the *exact* inputs used by each optimisation.
    """
    cov_dates = sorted(cov_dict.keys())
    holdings = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
    diagnostics: list[dict] = []

    for dt in alpha.index:
        a = alpha.loc[dt].dropna()
        if len(a) < 2:
            continue
        valid = [d for d in cov_dates if d <= dt]
        if not valid:
            continue
        cov = cov_dict[valid[-1]]
        dur_idx = duration.index[duration.index <= dt]
        if len(dur_idx) == 0:
            continue
        dur = duration.loc[dur_idx[-1]]
        common = a.index.intersection(cov.columns).intersection(dur.index)
        if len(common) < 2:
            continue

        h, status = optimize_portfolio(
            a[common],
            cov.loc[common, common],
            dur[common],
            risk_aversion,
            vol_ceiling,
            max_gross,
            max_position,
        )
        holdings.loc[dt, common] = h.values

        # Collect constraint diagnostics from exact optimizer inputs
        w_arr = h.values
        dur_arr = dur[common].values.astype(float)
        cov_arr = cov.loc[common, common].values.astype(float)
        dur_residual = float(dur_arr @ w_arr)
        gross_dur = float(np.abs(dur_arr) @ np.abs(w_arr))
        diagnostics.append(
            {
                "date": dt,
                "solver_status": status,
                "duration_residual": dur_residual,
                "gross_duration": gross_dur,
                "gross_exposure": float(np.abs(w_arr).sum()),
                "max_position": float(np.abs(w_arr).max()) if np.abs(w_arr).max() > 0 else 0.0,
                "ex_ante_var": float(w_arr @ cov_arr @ w_arr),
            }
        )

    diag_df = pd.DataFrame(diagnostics)
    if not diag_df.empty:
        diag_df = diag_df.set_index("date")

    return holdings, diag_df


def full_backtest(
    score,
    cov_dict,
    ret,
    duration,
    ic=0.05,
    risk_aversion=1.0,
    vol_ceiling=0.10,
    max_gross=2.0,
    max_position=0.5,
):
    """End-to-end: alpha + optimise + performance stats.

    Returns (holdings, alpha, EmpPerfStats, diagnostics_df).
    """
    from .evaluation import EmpPerfStats  # local import to avoid circular dep

    alpha = compute_alpha(score, cov_dict, ic=ic)
    holdings, diag_df = run_backtest(
        alpha, cov_dict, ret, duration, risk_aversion, vol_ceiling, max_gross, max_position
    )
    h_a, r_a = univ_align(holdings, ret)
    return holdings, alpha, EmpPerfStats(h_a, r_a), diag_df
