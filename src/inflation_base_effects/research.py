"""Auditable research calculations shared by the three notebooks.

Returns are decimal synthetic proxies. No helper selects a winning model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.regression.linear_model import OLS

from .data import compute_base_effect
from .evaluation import hac_mean_inference, portfolio_returns
from .portfolio import compute_bond_returns, ewmacov, full_backtest
from .signals import calc_signal_from_base_effect, univ_align


def mechanical_example(offset: bool = False) -> pd.DataFrame:
    """A 1% price increase at -12; optionally another proportional 1% at 0.

    Extra prehistory permits exact ordinary YoY percentage changes throughout
    the displayed range. Month 0 is the first increase's departure, not a fit.
    """
    months = pd.Index(range(-36, 14), name="month relative to departure")
    price = pd.Series(100.0, index=months)
    price.loc[-12:] *= 1.01
    if offset:
        price.loc[0:] *= 1.01
    yoy = 100 * (price / price.shift(12) - 1)
    return pd.DataFrame({"price": price, "yoy_pct": yoy, "change_yoy_pp": yoy.diff()}).loc[-15:13]


def plot_mechanical_effect():
    """Aligned price, YoY, and timing bars; two deterministic scenarios."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    for column, offset in enumerate([False, True]):
        data = mechanical_example(offset)
        axes[0, column].step(data.index, data.price, where="post", color="#234e70")
        axes[1, column].step(data.index, data.yoy_pct, where="post", color="#21867a")
        axes[2, column].bar(
            data.index,
            data.change_yoy_pp,
            color=np.where(data.change_yoy_pp >= 0, "#21867a", "#c75643"),
        )
        axes[0, column].set_title(
            "A. One increase, then unchanged prices"
            if not offset
            else "B. A new 1% increase offsets the departure"
        )
        for row in range(3):
            ax = axes[row, column]
            ax.axvline(-12, color="gray", ls=":", lw=1)
            ax.axvline(0, color="black", ls="--", lw=1)
            ax.set_xticks([-12, -6, 0, 6, 12])
        axes[2, column].axhline(0, color="black", lw=0.8)
        axes[2, column].set_xlabel("Months relative to first increase leaving YoY window")
    axes[0, 0].set_ylabel("CPI price level")
    axes[1, 0].set_ylabel("YoY inflation (%)")
    axes[2, 0].set_ylabel("Change in YoY inflation (pp)")
    axes[0, 0].annotate(
        "Prices do not fall", xy=(2, 101), xytext=(1, 100.55), arrowprops={"arrowstyle": "->"}
    )
    axes[2, 0].annotate(
        "Old increase exits: -1 pp",
        xy=(0, -1),
        xytext=(1, -0.6),
        arrowprops={"arrowstyle": "->"},
        fontsize=9,
    )
    fig.suptitle("Inflation base effects: accounting illustration, not estimated lead-lag evidence")
    return fig, axes


def coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, series in frame.items():
        valid = series.dropna()
        rows[name] = {
            "first": valid.index.min(),
            "last": valid.index.max(),
            "N": len(valid),
            "missing_pct": 100 * series.isna().mean(),
        }
    return pd.DataFrame(rows).T


def mechanism_regression(base: pd.Series, revision: pd.Series) -> pd.Series:
    """OLS in percentage-point units with automatic HAC uncertainty."""
    # Two distinct SPF surveys can have deadlines in the same calendar month.
    # Preserve each survey observation; never aggregate or duplicate via a join.
    frame = pd.DataFrame(
        {"base": base.reindex(revision.index).to_numpy(), "revision": revision.to_numpy()},
        index=revision.index,
    ).dropna()
    if len(frame) < 3:
        raise ValueError("mechanism regression requires at least three observations")
    lags = max(int(np.floor(4 * (len(frame) / 100) ** (2 / 9))), 1)
    fit = OLS(frame.revision, pd.DataFrame({"constant": 1.0, "base": frame.base})).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    ci = fit.conf_int().loc["base"]
    return pd.Series(
        {
            "N": len(frame),
            "HAC lags": lags,
            "beta": fit.params["base"],
            "CI lower": ci.iloc[0],
            "CI upper": ci.iloc[1],
            "HAC t": fit.tvalues["base"],
            "p": fit.pvalues["base"],
            "R2": fit.rsquared,
        }
    )


def realized_turnover(holdings: pd.DataFrame) -> pd.Series:
    """One-way absolute notional changes, charged to the next earning month.

    Initial funding is included; the final liquidation is not. There is no
    wealth drift adjustment: holdings are fixed reference-notional weights.
    """
    changes = holdings.diff()
    if not holdings.empty:
        changes.iloc[0] = holdings.iloc[0]
    return changes.abs().sum(axis=1, min_count=len(holdings.columns)).shift(1)


def performance_summary(
    returns: pd.Series, turnover: pd.Series | None = None, maxlags: int | None = None
) -> pd.Series:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    stats = hac_mean_inference(clean, maxlags=maxlags)
    # Reference-notional cumulative P&L is additive, not a funded wealth index.
    pnl = clean.cumsum()
    drawdown = pnl - pnl.cummax().clip(lower=0)
    stats["max_additive_drawdown"] = drawdown.min()
    stats["mean_monthly_turnover"] = (
        turnover.reindex(clean.index).mean() if turnover is not None else np.nan
    )
    stats["start"] = clean.index.min().strftime("%Y-%m")
    stats["end"] = clean.index.max().strftime("%Y-%m")
    return stats


def comparison_table(
    returns: dict[str, pd.Series], holdings: dict | None = None, common: bool = True
) -> pd.DataFrame:
    aligned = pd.concat(returns, axis=1).dropna() if common else None
    rows = {}
    for label, series in returns.items():
        turnover = realized_turnover(holdings[label]) if holdings is not None else None
        rows[label] = performance_summary(aligned[label] if common else series, turnover)
    return pd.DataFrame(rows).T


def return_components(yields: pd.DataFrame, duration: pd.DataFrame | None = None):
    if duration is None:
        _, duration = compute_bond_returns(yields)
    carry = yields.shift(1) / 1200
    price = -duration.shift(1) * yields.diff() / 100
    return carry, price


def portfolio_attribution(holdings: pd.DataFrame, yields: pd.DataFrame):
    returns, duration = compute_bond_returns(yields)
    h, r = univ_align(holdings, returns)
    prior = h.shift(1)
    valid = prior.notna().all(axis=1) & r.notna().all(axis=1)
    country = prior.loc[valid] * r.loc[valid]
    carry, price = return_components(yields, duration)
    components = pd.DataFrame(
        {
            "carry": (prior.loc[valid] * carry.loc[valid.index[valid], h.columns]).sum(axis=1),
            "yield_change": (prior.loc[valid] * price.loc[valid.index[valid], h.columns]).sum(
                axis=1
            ),
        }
    )
    # Ex-post descriptive decomposition; sample-average holdings are not a strategy.
    tilt = (prior.loc[valid].mean() * r.loc[valid]).sum(axis=1)
    components["total"] = country.sum(axis=1)
    components["tilt"] = tilt
    components["timing"] = components.total - tilt
    return country, components


def benchmark_regression(strategy: pd.Series, benchmark: pd.Series) -> pd.Series:
    frame = pd.concat({"strategy": strategy, "benchmark": benchmark}, axis=1).dropna()
    if len(frame) < 3 or frame.benchmark.std() == 0:
        raise ValueError("benchmark regression requires variation and at least three observations")
    lags = max(int(np.floor(4 * (len(frame) / 100) ** (2 / 9))), 1)
    fit = OLS(frame.strategy, pd.DataFrame({"constant": 1.0, "beta": frame.benchmark})).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    ci = fit.conf_int().loc["constant"] * 12
    return pd.Series(
        {
            "N": len(frame),
            "ann_intercept": fit.params["constant"] * 12,
            "CI lower": ci.iloc[0],
            "CI upper": ci.iloc[1],
            "HAC t": fit.tvalues["constant"],
            "p": fit.pvalues["constant"],
            "beta": fit.params["beta"],
            "correlation": frame.corr().iloc[0, 1],
            "R2": fit.rsquared,
            "HAC lags": lags,
        }
    )


def constraint_summary(diagnostics: pd.DataFrame) -> pd.Series:
    d = diagnostics
    return pd.Series(
        {
            "months": len(d),
            "failed": (~d.solver_status.isin(["optimal", "optimal_inaccurate"])).sum(),
            "max_duration_residual": d.duration_residual.abs().max(),
            "max_gross": d.gross_exposure.max(),
            "max_position": d.max_position.max(),
            "max_forecast_vol": np.sqrt(12 * d.ex_ante_var.max()),
            "position_binding_pct": 100 * (d.max_position >= 0.5 - 1e-3).mean(),
            "gross_binding_pct": 100 * (d.gross_exposure >= 2 - 1e-3).mean(),
            "vol_binding_pct": 100 * (np.sqrt(12 * d.ex_ante_var) >= 0.10 - 1e-3).mean(),
        }
    )


@dataclass
class Experiment:
    holdings: pd.DataFrame
    returns: pd.Series
    diagnostics: pd.DataFrame


def run_experiment(
    cpi: pd.DataFrame,
    yields: pd.DataFrame,
    *,
    hl: int = 6,
    implag: int = 1,
    ic: float = 0.05,
    covariance_halflives: tuple[int, int] = (12, 36),
    legacy_duration: bool = False,
) -> Experiment:
    """Same primary model, with explicit one-at-a-time experimental controls."""
    returns, duration = compute_bond_returns(yields)
    if legacy_duration:
        duration = yields.map(
            lambda x: 10.0 if pd.isna(x) or x <= 0.1 else (1 - (1 + x / 200) ** -20) / (x / 100)
        )
        carry, price = return_components(yields, duration)
        returns = (carry + price).dropna(how="all")
    score = calc_signal_from_base_effect(
        compute_base_effect(cpi), returns.index[-1], hl=hl, implag=implag
    ).dropna(how="any")
    covariance = ewmacov(returns.dropna(), halflife=covariance_halflives, shrinkage_factor=0.3)
    h, _, perf, diag = full_backtest(score, covariance, returns, duration, ic=ic)
    return Experiment(h, perf.port_ret, diag)


def leave_one_out(cpi: pd.DataFrame, yields: pd.DataFrame) -> dict[str, Experiment]:
    """Recompute scores, covariance, alpha, and optimization on each reduced universe."""
    return {
        f"Without {country}": run_experiment(
            cpi.drop(columns=country), yields.drop(columns=country)
        )
        for country in cpi.columns.intersection(yields.columns)
    }


def sensitivity_experiments(
    cpi: pd.DataFrame, yields: pd.DataFrame, baseline: Experiment | None = None
):
    """Declared one-at-a-time grid, reusing the baseline (never ranking/selecting)."""
    baseline = baseline or run_experiment(cpi, yields)
    families = {
        "Smoothing": ("hl", [3, 6, 12], 6),
        "Assumed IC": ("ic", [0.01, 0.025, 0.05, 0.10], 0.05),
        "Implementation delay": ("implag", [1, 2, 3], 1),
        "Covariance": ("covariance_halflives", [(6, 18), (12, 36), (24, 72)], (12, 36)),
    }
    groups = {}
    for name, (parameter, values, default) in families.items():
        groups[name] = {
            str(value): baseline
            if value == default
            else run_experiment(cpi, yields, **{parameter: value})
            for value in values
        }
    return groups


def legacy_evaluation_comparison(holdings: pd.DataFrame, yields: pd.DataFrame):
    """Isolate evaluation conventions with identical already-formed holdings.

    These are diagnostics, not an exact historical reconstruction. The
    contemporaneous row is explicitly noncausal and never used as a strategy.
    """
    ret, _ = compute_bond_returns(yields)
    h, r = univ_align(holdings, ret)
    return {
        "Prior holdings, complete cases": portfolio_returns(h, r),
        "Prior holdings, missing contributions zero": (h.shift(1) * r).sum(axis=1),
        "Contemporaneous holdings (NONCAUSAL)": (h * r).sum(axis=1),
    }
