"""Backtest evaluation: performance statistics, lead/lag, tilt/timing, additivity, plotting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.regression.linear_model import OLS

from .signals import univ_align

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_ppy(idx: pd.DatetimeIndex) -> int:
    """Infer periods-per-year from a DatetimeIndex."""
    if len(idx) < 2:
        return 12
    gap = (idx[-1] - idx[0]).days / max(len(idx) - 1, 1)
    if gap < 3:
        return 252
    if gap < 10:
        return 52
    if gap < 45:
        return 12
    return 4


def _newey_west_lags(sample_size: int) -> int:
    """Return the same automatic HAC lag rule used throughout the study."""
    return max(int(np.floor(4 * (sample_size / 100) ** (2 / 9))), 1)


def _clean_return_series(returns: pd.Series) -> pd.Series:
    """Coerce a return series to finite floats and remove missing observations."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    cleaned = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")
    return cleaned.astype(float).sort_index()


def hac_mean_inference(
    returns: pd.Series,
    periods_per_year: int = 12,
    maxlags: int | None = None,
) -> pd.Series:
    """Estimate an annualized mean return with Newey-West inference.

    Missing and non-finite observations are excluded. The confidence interval
    and standard error are scaled to annual units; the t-statistic and p-value
    test whether the unannualized mean is zero.
    """
    clean = _clean_return_series(returns)
    if len(clean) < 3:
        raise ValueError("at least three finite return observations are required")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    lags = _newey_west_lags(len(clean)) if maxlags is None else maxlags
    if not isinstance(lags, int) or lags < 0:
        raise ValueError("maxlags must be a non-negative integer")

    result = OLS(clean.to_numpy(), np.ones((len(clean), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    mean = float(clean.mean())
    vol = float(clean.std() * np.sqrt(periods_per_year))
    annualized_mean = mean * periods_per_year
    annualized_se = float(result.bse[0] * periods_per_year)
    return pd.Series(
        {
            "N": len(clean),
            "ann_return": annualized_mean,
            "ann_vol": vol,
            "return_to_vol": annualized_mean / vol if vol > 0 else np.nan,
            "NW t-stat": float(result.tvalues[0]),
            "NW p-value": float(result.pvalues[0]),
            "CI 95% lower": annualized_mean - 1.96 * annualized_se,
            "CI 95% upper": annualized_mean + 1.96 * annualized_se,
            "NW maxlags": lags,
        },
        dtype=float,
    )


def regime_stability(
    returns: pd.Series,
    split_date: str | pd.Timestamp,
    periods_per_year: int = 12,
    maxlags: int | None = None,
) -> pd.DataFrame:
    """Compare return performance before and after a fixed date using HAC inference.

    The ``post_minus_pre`` row is the annualized coefficient on a post-period
    indicator in a full-sample regression. At least twelve observations are
    required in each regime so very short segments cannot be overinterpreted.
    """
    clean = _clean_return_series(returns)
    split = pd.Timestamp(split_date)
    pre = clean.loc[clean.index < split]
    post = clean.loc[clean.index >= split]
    if len(pre) < 12 or len(post) < 12:
        raise ValueError("split_date must leave at least twelve observations in each regime")

    pre_stats = hac_mean_inference(pre, periods_per_year, maxlags)
    post_stats = hac_mean_inference(post, periods_per_year, maxlags)
    lags = _newey_west_lags(len(clean)) if maxlags is None else maxlags
    if not isinstance(lags, int) or lags < 0:
        raise ValueError("maxlags must be a non-negative integer")
    post_indicator = (clean.index >= split).astype(float)
    design = np.column_stack([np.ones(len(clean)), post_indicator])
    result = OLS(clean.to_numpy(), design).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    annualized_change = float(result.params[1] * periods_per_year)
    annualized_se = float(result.bse[1] * periods_per_year)
    change = pd.Series(
        {
            "N": len(clean),
            "ann_return": annualized_change,
            "ann_vol": np.nan,
            "return_to_vol": np.nan,
            "NW t-stat": float(result.tvalues[1]),
            "NW p-value": float(result.pvalues[1]),
            "CI 95% lower": annualized_change - 1.96 * annualized_se,
            "CI 95% upper": annualized_change + 1.96 * annualized_se,
            "NW maxlags": lags,
        },
        dtype=float,
    )
    return pd.DataFrame(
        [pre_stats, post_stats, change],
        index=["pre", "post", "post_minus_pre"],
    )


def rolling_hac_mean(
    returns: pd.Series,
    window: int = 120,
    periods_per_year: int = 12,
    maxlags: int | None = None,
) -> pd.DataFrame:
    """Compute trailing-window annualized means and Newey-West confidence intervals."""
    clean = _clean_return_series(returns)
    if not isinstance(window, int) or window < 3:
        raise ValueError("window must be an integer of at least three observations")
    if len(clean) < window:
        raise ValueError("return history is shorter than the requested rolling window")

    rows = []
    for end in range(window, len(clean) + 1):
        stats = hac_mean_inference(
            clean.iloc[end - window : end],
            periods_per_year=periods_per_year,
            maxlags=maxlags,
        )
        rows.append(stats)
    result = pd.DataFrame(rows, index=clean.index[window - 1 :])
    numeric = result.to_numpy(dtype=float)
    if not np.isfinite(numeric[:, [0, 1, 2, 4, 5, 6, 7, 8]]).all():
        raise RuntimeError("rolling HAC calculation produced a non-finite required statistic")
    return result


def portfolio_returns(
    holdings: pd.DataFrame, returns: pd.DataFrame, holding_lag: int = 1
) -> pd.Series:
    """Compute complete-case portfolio returns with explicit holding timing.

    A date is excluded if any required prior holding or contemporaneous asset
    return is unavailable.  Missing contributions are never converted to zero.
    """
    holdings, returns = univ_align(holdings, returns)
    lagged = holdings.shift(holding_lag)
    valid = lagged.notna().all(axis=1) & returns.notna().all(axis=1)
    contributions = lagged.loc[valid] * returns.loc[valid]
    return contributions.sum(axis=1, min_count=len(contributions.columns))


def plotgrid(data: pd.DataFrame, title: str | None = None, figsize=None):
    """Plot a MultiIndex or flat DataFrame as a grid of subplots."""
    if data.empty:
        return
    if isinstance(data.columns, pd.MultiIndex):
        groups = data.columns.get_level_values(0).unique()
        assets = data.columns.get_level_values(1).unique()
        if figsize is None:
            figsize = (15, 3.5 * len(assets))
        fig, axes = plt.subplots(
            len(assets), len(groups), figsize=figsize, sharex=True, squeeze=False
        )
        for i, asset in enumerate(assets):
            for j, group in enumerate(groups):
                ax = axes[i, j]
                key = (group, asset)
                if key in data.columns:
                    s = data[key].dropna()
                    if not s.empty:
                        ax.plot(s.index, s.values, linewidth=0.8)
                ax.set_title(f"{group} - {asset}", fontsize=9)
                ax.tick_params(axis="x", rotation=45, labelsize=7)
                ax.tick_params(axis="y", labelsize=7)
                ax.axhline(0, color="gray", ls="--", alpha=0.4, lw=0.5)
    else:
        n = len(data.columns)
        if figsize is None:
            figsize = (15, 3 * n)
        fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, squeeze=False)
        for i, col in enumerate(data.columns):
            s = data[col].dropna()
            if not s.empty:
                axes[i, 0].plot(s.index, s.values, linewidth=0.8)
            axes[i, 0].set_title(col, fontsize=9)
            axes[i, 0].tick_params(axis="x", rotation=45, labelsize=7)
            axes[i, 0].axhline(0, color="gray", ls="--", alpha=0.4, lw=0.5)
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Empirics classes
# ---------------------------------------------------------------------------


class EmpPerfStats:
    """Annualised return, vol, IR (full / split-sample), max drawdown."""

    def __init__(self, h: pd.DataFrame, ret: pd.DataFrame):
        h, ret = univ_align(h, ret)
        self.h, self.ret = h, ret
        self.ppy = _infer_ppy(ret.index)
        self.port_ret = portfolio_returns(h, ret, holding_lag=1)
        self.VA = self.port_ret.cumsum()
        self.ann_ret = self.port_ret.mean() * self.ppy
        self.ann_vol = self.port_ret.std() * np.sqrt(self.ppy)
        self.IR = self.ann_ret / self.ann_vol if self.ann_vol > 0 else 0.0
        mid = len(self.port_ret) // 2
        r1, r2 = self.port_ret.iloc[:mid], self.port_ret.iloc[mid:]
        self.IR_H1 = (r1.mean() / r1.std()) * np.sqrt(self.ppy) if r1.std() > 0 else 0.0
        self.IR_H2 = (r2.mean() / r2.std()) * np.sqrt(self.ppy) if r2.std() > 0 else 0.0
        cumret = (1 + self.port_ret).cumprod()
        self.max_dd = (cumret / cumret.cummax() - 1).min()

    def plot(self, figsize=(15, 5)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self.VA.plot(ax=axes[0], title=f"Cumulative VA (IR={self.IR:.2f})")
        axes[0].axhline(0, color="gray", ls="--", alpha=0.5)
        axes[0].set_ylabel("Cumulative Return")
        w = max(12, len(self.port_ret) // 8)
        roll = (self.port_ret.rolling(w).mean() / self.port_ret.rolling(w).std()) * np.sqrt(
            self.ppy
        )
        roll.plot(ax=axes[1], title=f"Rolling IR (window={w})")
        axes[1].axhline(0, color="gray", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
        print(
            f"Ann. Ret: {self.ann_ret:.4f} | Ann. Vol: {self.ann_vol:.4f} | "
            f"IR: {self.IR:.2f} | H1: {self.IR_H1:.2f} | H2: {self.IR_H2:.2f} | "
            f"Max DD: {self.max_dd:.2%}"
        )


class EmpLeadLag:
    """Lead/lag IR analysis."""

    def __init__(self, h: pd.DataFrame, ret: pd.DataFrame, leadlags=None):
        h, ret = univ_align(h, ret)
        self.leadlags = leadlags or range(-6, 13)
        ppy = _infer_ppy(ret.index)
        self.IRs: dict[int, float] = {}
        self.lag_va: dict[int, pd.Series] = {}
        for lag in self.leadlags:
            pr = portfolio_returns(h, ret, holding_lag=lag)
            self.IRs[lag] = (
                (pr.mean() / pr.std()) * np.sqrt(ppy) if len(pr) > 10 and pr.std() > 0 else 0.0
            )
            self.lag_va[lag] = pr.cumsum()

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        irs = list(self.IRs.values())
        ax.bar(
            self.IRs.keys(),
            irs,
            color=["steelblue" if v > 0 else "salmon" for v in irs],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel("Lead / Lag")
        ax.set_ylabel("IR")
        ax.set_title("Lead/Lag IR")
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        # Annotate the lag=0 bar with its IR value
        if 0 in self.IRs:
            ir0 = self.IRs[0]
            ax.annotate(
                f"IR={ir0:.2f}",
                xy=(0, ir0),
                xytext=(0, ir0 + 0.05 * (1 if ir0 >= 0 else -1)),
                ha="center",
                fontsize=8,
                color="black",
            )
        # Color legend
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(color="steelblue", alpha=0.8, label="Positive IR"),
                Patch(color="salmon", alpha=0.8, label="Negative IR"),
            ],
            fontsize=8,
            loc="upper right",
        )

    def plot_lagperf(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        for lag, va in self.lag_va.items():
            ax.plot(
                va.index,
                va.values,
                alpha=0.3 + 0.7 * (lag == 0),
                lw=0.5 + 1.5 * (lag == 0),
                label=str(lag) if lag in (0, min(self.leadlags), max(self.leadlags)) else None,
            )
        ax.legend(fontsize=8, title="Lag")
        ax.set_title("Cumulative Performance by Lag")
        ax.axhline(0, color="gray", ls="--", alpha=0.5)


class EmpTiltTiming:
    """Tilt / timing decomposition using prior-period holdings."""

    def __init__(self, h: pd.DataFrame, ret: pd.DataFrame):
        h, ret = univ_align(h, ret)
        lagged = h.shift(1)
        valid = lagged.notna().all(axis=1) & ret.notna().all(axis=1)
        h = lagged.loc[valid]
        ret = ret.loc[valid]
        ppy = _infer_ppy(ret.index)
        avg = h.mean()
        tilt = (avg * ret).sum(axis=1)
        timing = (h.sub(avg) * ret).sum(axis=1)
        total = (h * ret).sum(axis=1)

        def annualized_ratio(series):
            return (series.mean() / series.std()) * np.sqrt(ppy) if series.std() > 0 else 0.0

        self.tilt_ir = annualized_ratio(tilt)
        self.timing_ir = annualized_ratio(timing)
        self.total_ir = annualized_ratio(total)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        labels = ["Total", "Tilt", "Timing"]
        values = [self.total_ir, self.tilt_ir, self.timing_ir]
        colors = ["steelblue", "seagreen", "orange"]
        bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="white")
        # Add IR value labels above each bar
        for bar, val in zip(bars, values, strict=True):
            y = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{val:.2f}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_ylabel("IR")
        ax.set_title("Tilt / Timing Decomposition")
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        # Color legend
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(color="steelblue", alpha=0.8, label="Total IR"),
                Patch(color="seagreen", alpha=0.8, label="Tilt IR (avg holdings)"),
                Patch(color="orange", alpha=0.8, label="Timing IR (dynamic)"),
            ],
            fontsize=7,
            loc="best",
        )


class EmpAdditivity:
    """Signal additivity analysis."""

    def __init__(self, vas: pd.DataFrame, new_signame: str):
        self.vas = vas
        self.new_signame = new_signame
        self.rets = vas.diff().dropna()
        self.combined_ret = self.rets.mean(axis=1)
        self.combined_va = self.combined_ret.cumsum()
        ppy = _infer_ppy(vas.index)

        def annualized_ratio(series):
            return (series.mean() / series.std()) * np.sqrt(ppy) if series.std() > 0 else 0.0

        self.ir_dict = {c: annualized_ratio(self.rets[c]) for c in self.rets.columns}
        self.ir_combined = annualized_ratio(self.combined_ret)
        self.corr = self.rets.corr()

    def plot(self, figsize=(12, 5)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        labels = list(self.ir_dict.keys()) + ["Combined"]
        vals = list(self.ir_dict.values()) + [self.ir_combined]
        axes[0].bar(
            labels, vals, color=["steelblue"] * len(self.ir_dict) + ["darkgreen"], alpha=0.8
        )
        axes[0].set_title("Individual vs Combined IR")
        axes[0].set_ylabel("IR")
        axes[0].axhline(0, color="gray", ls="--", alpha=0.5)
        im = axes[1].imshow(self.corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axes[1].set_xticks(range(len(self.corr)))
        axes[1].set_yticks(range(len(self.corr)))
        axes[1].set_xticklabels(self.corr.columns, rotation=45, ha="right")
        axes[1].set_yticklabels(self.corr.columns)
        axes[1].set_title("Return Correlation")
        plt.colorbar(im, ax=axes[1], shrink=0.8)
        plt.tight_layout()
        plt.show()

    def plot_additivity_perf(self, figsize=(15, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        for c in self.vas.columns:
            ax.plot(self.vas.index, self.vas[c], label=c)
        ax.plot(
            self.combined_va.index, self.combined_va, label="Combined", lw=2, ls="--", color="black"
        )
        ax.legend()
        ax.set_title("Cumulative Performance: Individual vs Combined")
        ax.set_ylabel("Cumulative Return")
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
