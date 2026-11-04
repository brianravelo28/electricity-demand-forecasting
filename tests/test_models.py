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


def test_recent_week_forecast_is_missing_where_the_level_is_missing():
    df = frame()
    train, test = df.iloc[:5], df.iloc[4:]
    pred, _ = models.fit_predict_lgbm(train, test, "A", ["X", "Y"], rounds=3, scaling="recent_week")
    assert np.isfinite(pred[0]) and np.isnan(pred[1])
