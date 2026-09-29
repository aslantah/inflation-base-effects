"""Signal construction utilities: scoring, alignment, base-effects signal."""

from __future__ import annotations

import numpy as np
import pandas as pd


def tscore(x: pd.DataFrame, halflife: int = 6) -> pd.DataFrame:
    """Time-series z-score with exponentially weighted mean and std."""
    min_periods = max(halflife, 3)
    mu = x.ewm(halflife=halflife, min_periods=min_periods).mean()
    sigma = x.ewm(halflife=halflife, min_periods=min_periods).std()
    return (x - mu) / sigma


def expanding_month_of_year_zscore(x: pd.DataFrame, min_history: int = 5) -> pd.DataFrame:
    """Normalize each calendar month using only prior same-month observations.

    The current observation is excluded from its own mean and standard deviation.
    This makes the transformation strictly expanding and real-time safe.
    """
    if min_history < 2:
        raise ValueError("min_history must be at least 2")
    result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for month in range(1, 13):
        month_values = x.loc[x.index.month == month]
        history = month_values.shift(1)
        mean = history.expanding(min_periods=min_history).mean()
        std = history.expanding(min_periods=min_history).std()
        result.loc[month_values.index] = (month_values - mean) / std.replace(0, np.nan)
    return result.sort_index()


def xscore(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score at each time step."""
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


def from_2d(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a MultiIndex DataFrame from a dict of DataFrames."""
    return pd.concat(d, axis=1)


def univ_align(*dfs: pd.DataFrame):
    """Align DataFrames on common index and columns.

    Returns a single DataFrame when given one argument, otherwise a tuple.
    """
    idx = dfs[0].index
    cols = dfs[0].columns
    for df in dfs[1:]:
        idx = idx.intersection(df.index)
        cols = cols.intersection(df.columns)
    result = tuple(df.loc[idx, cols] for df in dfs)
    return result[0] if len(result) == 1 else result


def calc_signal_from_base_effect(
    rolling_off: pd.DataFrame,
    last_return_date: pd.Timestamp,
    hl: int = 6,
    implag: int = 1,
) -> pd.DataFrame:
    """Construct the base-effects trading signal from the raw rolling-off print.

    The input ``rolling_off`` is already lagged by 12 months (computed by
    ``compute_base_effect``), so no additional shift is applied here.

    Parameters
    ----------
    rolling_off : DataFrame
        Monthly log-inflation prints rolling off the YoY window, as returned
        by ``compute_base_effect``.  Already shifted by 12 months.
    last_return_date : Timestamp
        Last date for which bond returns are available.
    hl : int
        Halflife for time-series z-score (months).
    implag : int
        Implementation lag (months).
    """
    score = tscore(rolling_off, halflife=hl)
    score = score.clip(-2, 2)
    score = score.resample("MS").last().shift(implag).dropna(how="all")
    score = score.loc[:last_return_date]
    return score


def calc_month_of_year_signal(
    rolling_off: pd.DataFrame,
    last_return_date: pd.Timestamp,
    min_history: int = 5,
    implag: int = 1,
) -> pd.DataFrame:
    """Construct a real-time seasonal robustness signal from NSA base effects."""
    score = expanding_month_of_year_zscore(rolling_off, min_history=min_history)
    score = score.clip(-2, 2)
    score = score.resample("MS").last().shift(implag).dropna(how="all")
    return score.loc[:last_return_date]


def calc_signal(
    cpi_mom_sa: pd.DataFrame,
    last_return_date: pd.Timestamp,
    shiftval: int = 12,
    hl: int = 6,
    implag: int = 1,
) -> pd.DataFrame:
    """Construct the base-effects signal from seasonally adjusted MoM CPI.

    This version uses full-sample seasonal adjustment and is provided as a
    robustness check.  The primary signal should use ``calc_signal_from_base_effect``.

    Parameters
    ----------
    cpi_mom_sa : DataFrame
        Seasonally adjusted MoM CPI (annualised).
    last_return_date : Timestamp
        Last date for which bond returns are available.
    shiftval : int
        Months to shift CPI (12 = base effect rolling off the YoY window).
    hl : int
        Halflife for time-series z-score (months).
    implag : int
        Implementation lag (months).
    """
    sig = cpi_mom_sa.shift(shiftval, freq="MS")
    score = tscore(sig, halflife=hl)
    score = score.clip(-2, 2)
    score = score.resample("MS").last().shift(implag).dropna(how="all")
    score = score.loc[:last_return_date]
    return score
