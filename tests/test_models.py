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


def test_the_recent_week_forecast_is_missing_where_the_level_is_missing():
    df = frame()
    train, test = df.iloc[:5], df.iloc[4:]
    pred, _ = models.fit_predict_lgbm(train, test, "A", ["X", "Y"], rounds=3, scaling="recent_week")
    assert np.isfinite(pred[0]) and np.isnan(pred[1])
