import numpy as np
import pandas as pd
import pytest

from electricity import models


def frame():
    cols = models.feature_columns("A")
    df = pd.DataFrame({c: np.arange(1.0, 7.0) for c in cols})
    df["ba"] = ["X", "X", "X", "Y", "Y", "Y"]
    df["y"] = [100.0, 110.0, 120.0, 10.0, 11.0, 12.0]
    df["demand_mean_168h"] = [100.0, 100.0, 100.0, 10.0, 10.0, np.nan]
    df["lag_day"] = [100.0, 110.0, 120.0, 10.0, 11.0, 12.0]
    return df


def test_default_scales_by_the_training_mean_of_each_ba():
    df = frame()
    scale = models.training_scale(df)
    x, s = models._matrix(df, scale, ["X", "Y"], "A")
    assert list(s) == [110.0, 110.0, 110.0, 11.0, 11.0, 11.0]
    assert list(x["lag_day"]) == pytest.approx(list(df["lag_day"] / s))


def test_recent_week_scales_by_the_trailing_week_mean():
    df = frame()
    x, s = models._matrix(df, models.training_scale(df), ["X", "Y"], "A", "recent_week")
    assert list(s[:5]) == [100.0, 100.0, 100.0, 10.0, 10.0]
    assert np.isnan(s[5])
    assert x["lag_day"].iloc[1] == pytest.approx(1.1)
    assert x["demand_mean_168h"].iloc[0] == pytest.approx(1.0)  # scaled by itself


def test_an_unknown_scaling_is_refused():
    df = frame()
    with pytest.raises(ValueError):
        models._matrix(df, models.training_scale(df), ["X", "Y"], "A", "nope")


def synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    cols = models.feature_columns("A")
    df = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    level = 100 + 20 * rng.random(n)
    for c in models.LEVEL_COLS:
        df[c] = level * (1 + 0.05 * rng.normal(size=n))
    df["ba"] = rng.choice(["X", "Y"], n)
    df["y"] = df["lag_day"] * (1 + 0.03 * df["temp_last"]) + rng.normal(0, 1, n)
    return df


def retrain(df, rounds=20):
    return models.fit_predict_lgbm(df.iloc[:3000], df.iloc[3000:], "A", ["X", "Y"], rounds)[0]


def test_a_retrain_with_the_same_seed_gives_identical_forecasts():
    df = synthetic()
    runs = [retrain(df) for _ in range(4)]
    assert all(np.array_equal(runs[0], r) for r in runs[1:])


def test_a_different_seed_gives_different_forecasts(monkeypatch):
    df = synthetic()
    first = retrain(df)
    monkeypatch.setitem(models.LGBM_PARAMS, "seed", 7)
    assert not np.array_equal(first, retrain(df))


def ridge_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    cols = models.feature_columns("B")  # the widest set of inputs
    df = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    level = 100 + 20 * rng.random(n)
    for c in models.LEVEL_COLS:
        df[c] = level * (1 + 0.02 * rng.normal(size=n))
    df["hour_local"] = rng.integers(0, 24, n)
    df["weekday"] = rng.integers(0, 7, n)
    df["month"] = rng.integers(1, 13, n)
    for c in ["is_holiday", "day_before_holiday", "day_after_holiday"]:
        df[c] = rng.integers(0, 2, n)
    df["lag_day_age"] = rng.integers(1, 3, n)
    df["horizon"] = rng.integers(14, 38, n)
    df["hours_in_day"] = 24
    df["ba"] = rng.choice(["X", "Y"], n)
    df["origin_date"] = pd.Timestamp("2022-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 400, n)), unit="D")
    df["y"] = 0.6 * df["lag_day"] + 0.4 * df["lag_week_1"] + rng.normal(0, 0.3, n)
    return df


def test_variant_n_has_no_weather_and_the_others_are_unchanged():
    n, a, b = (models.feature_columns(v) for v in "NAB")
    assert not set(n) & set(models.WEATHER_A + models.WEATHER_B)
    assert a == n + models.WEATHER_A and b == a + models.WEATHER_B


def test_ridge_recovers_a_linear_relation_on_unseen_rows():
    df = ridge_frame()
    train, test = df.iloc[:2400], df.iloc[2400:]
    pred = models.fit_predict_ridge(train, test, "A", ["X", "Y"])
    assert np.isfinite(pred).all()
    assert (100 * np.abs(pred - test["y"]) / test["y"]).mean() < 1.0


def test_ridge_forecast_is_missing_only_where_an_input_is_missing():
    df = ridge_frame()
    train, test = df.iloc[:2400], df.iloc[2400:].copy()
    test.iloc[0, test.columns.get_loc("temp_last")] = np.nan
    pred = models.fit_predict_ridge(train, test, "A", ["X", "Y"])
    assert np.isnan(pred[0]) and np.isfinite(pred[1:]).all()


def test_ridge_never_reads_the_test_targets_and_is_deterministic():
    df = ridge_frame()
    train, test = df.iloc[:2400], df.iloc[2400:].copy()
    first = models.fit_predict_ridge(train, test, "A", ["X", "Y"])
    test["y"] = -999.0
    assert np.array_equal(first, models.fit_predict_ridge(train, test, "A", ["X", "Y"]))


def test_ridge_and_lightgbm_run_without_any_weather_columns():
    df = ridge_frame().drop(columns=models.WEATHER_A + models.WEATHER_B)
    train, test = df.iloc[:2400], df.iloc[2400:]
    assert np.isfinite(models.fit_predict_ridge(train, test, "N", ["X", "Y"])).all()
    assert np.isfinite(models.fit_predict_lgbm(train, test, "N", ["X", "Y"], rounds=5)[0]).all()


def test_the_recent_week_forecast_is_missing_where_the_level_is_missing():
    df = frame()
    train, test = df.iloc[:5], df.iloc[4:]
    pred, _ = models.fit_predict_lgbm(train, test, "A", ["X", "Y"], rounds=3, scaling="recent_week")
    assert np.isfinite(pred[0]) and np.isnan(pred[1])
