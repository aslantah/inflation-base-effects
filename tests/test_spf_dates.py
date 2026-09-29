"""Verify parsing of official SPF deadline and release dates."""

import pandas as pd

from inflation_base_effects.data import parse_spf_release_dates

SAMPLE = """
1990 Q2             8/23/90*            8/31/90*
     Q3             8/23/90             8/31/90
2019 Q1             3/12/19**           3/22/19**
2026 Q1             3/2/26**            3/6/26**
"""


def test_spf_release_dates_parse_year_and_continuation_rows():
    dates = parse_spf_release_dates(SAMPLE)
    assert list(dates.index) == ["1990Q2", "1990Q3", "2019Q1", "2026Q1"]
    assert dates.loc["1990Q3", "deadline_date"] == pd.Timestamp("1990-08-23")


def test_spf_release_dates_preserve_delayed_first_quarter_surveys():
    dates = parse_spf_release_dates(SAMPLE)
    assert dates.loc["2019Q1", "release_date"] == pd.Timestamp("2019-03-22")
    assert dates.loc["2026Q1", "deadline_date"] == pd.Timestamp("2026-03-02")
