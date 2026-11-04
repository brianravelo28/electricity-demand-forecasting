import numpy as np
import pandas as pd

from electricity.cleaning import clean_demand

IDX = pd.date_range("2022-01-01", periods=24 * 40, freq="h", tz="UTC")


def base():
    hours = np.arange(len(IDX))
    return pd.Series(1000 + 200 * np.sin(2 * np.pi * hours / 24), index=IDX)


def test_clean_series_is_untouched():
    s = base()
    cleaned, bad = clean_demand(s)
    assert not bad.any()
    pd.testing.assert_series_equal(cleaned, s)


def test_impossible_values_are_removed():
    s = base()
    for pos, value in {500: 2_147_480_000.0, 600: -9_000_000.0, 700: 0.0, 800: 90.0, 850: 3000.0}.items():
        s.iloc[pos] = value
    cleaned, bad = clean_demand(s)
    assert set(np.flatnonzero(bad)) == {500, 600, 700, 800, 850}
    assert cleaned.iloc[[500, 600, 700, 800, 850]].isna().all()
    assert cleaned.drop(cleaned.index[[500, 600, 700, 800, 850]]).notna().all()


def test_a_run_of_bad_hours_is_removed():
    s = base()
    s.iloc[500:508] = 20_000.0
    _, bad = clean_demand(s)
    assert bad.iloc[500:508].all() and bad.sum() == 8


def test_missing_values_stay_missing_and_are_not_flagged():
    s = base()
    s.iloc[300] = np.nan
    cleaned, bad = clean_demand(s)
    assert np.isnan(cleaned.iloc[300]) and not bad.iloc[300]


def test_judgement_uses_only_earlier_readings():
    s = base()
    s.iloc[600] = 5_000_000.0
    _, bad = clean_demand(s)
    changed = s.copy()
    changed.iloc[601:] = 1.0  # rewrite everything after the reading
    _, bad_changed = clean_demand(changed)
    assert bad.iloc[:601].equals(bad_changed.iloc[:601])


def test_no_judgement_before_enough_history():
    s = base()
    s.iloc[10] = 5_000_000.0
    _, bad = clean_demand(s)
    assert not bad.iloc[10]
