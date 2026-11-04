"""Regression and LightGBM forecasts for the walk-forward evaluation (see docs/PLANNING.md).

Both models work on demand divided by each BA's mean demand over the training window, so a
fit never sees the test period. Predictions come back in MWh per hour.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from electricity.features import HDD_BASE_C

LEVEL_COLS = ["demand_last", "demand_mean_24h", "demand_max_24h", "demand_mean_168h", "demand_change_3h",
              "lag_week_1", "lag_week_2", "lag_week_3", "lag_week_4", "lag_day"]
CALENDAR_COLS = ["hour_local", "weekday", "month", "doy_sin", "doy_cos", "is_holiday", "day_before_holiday",
                 "day_after_holiday", "hours_in_day", "horizon", "lag_day_age"]
WEATHER_A = ["temp_last", "temp_mean_24h", "temp_change_3h", "dew_last", "wind_last"]
WEATHER_B = ["temp_target", "dew_target", "wind_target", "temp_day_max", "temp_day_min", "temp_day_mean",
             "hdd_day", "cdd_day"]
LGBM_PARAMS = {
    "objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "verbose": -1, "seed": 0, "num_threads": 0,
}


def feature_columns(variant):
    return LEVEL_COLS + CALENDAR_COLS + WEATHER_A + (WEATHER_B if variant == "B" else [])


def training_scale(train):
    """Mean demand per BA over the training window."""
    return train.dropna(subset=["y"]).groupby("ba")["y"].mean()


SCALINGS = ("training_mean", "recent_week")


def _matrix(df, scale, bas, variant, scaling="training_mean"):
    """Features with demand levels divided by a scale, and that scale per row.

    training_mean: each BA's mean demand over the training window (the default).
    recent_week:   the mean of the 168 hours up to the last demand reading, known at the origin,
                   so a BA whose demand has grown is scaled by where it is now.
    """
    x = df[feature_columns(variant)].astype(float)
    if scaling == "recent_week":
        s = df["demand_mean_168h"].to_numpy(dtype=float)
    elif scaling == "training_mean":
        s = df["ba"].map(scale).to_numpy(dtype=float)
    else:
        raise ValueError(f"scaling must be one of {SCALINGS}")
    for col in LEVEL_COLS:
        x[col] = df[col].to_numpy(dtype=float) / s
    x["ba"] = pd.Categorical(df["ba"], categories=list(bas))
    return x, s


def fit_predict_lgbm(train, test, variant, bas, rounds=400, scaling="training_mean"):
    """One global LightGBM across BAs. Rows whose scale is missing get a missing forecast."""
    scale = training_scale(train)
    train = train.dropna(subset=["y"])
    x_tr, s_tr = _matrix(train, scale, bas, variant, scaling)
    ok = np.isfinite(s_tr) & (s_tr > 0)
    data = lgb.Dataset(x_tr[ok], label=train["y"].to_numpy(dtype=float)[ok] / s_tr[ok], categorical_feature=["ba"])
    booster = lgb.train(LGBM_PARAMS, data, num_boost_round=rounds)
    x_te, s_te = _matrix(test, scale, bas, variant, scaling)
    return booster.predict(x_te) * s_te, booster


def _design(df, variant):
    """Hour-of-week effects, holiday flags and heating/cooling degrees by six-hour block."""
    n = len(df)
    how = (df["weekday"] * 24 + df["hour_local"]).to_numpy()
    week = np.zeros((n, 168))
    week[np.arange(n), how] = 1.0
    temp = df["temp_target"] if variant == "B" else df["temp_mean_24h"]
    temp = temp.to_numpy(dtype=float)
    hdd = np.maximum(0.0, HDD_BASE_C - temp)
    cdd = np.maximum(0.0, temp - HDD_BASE_C)
    block = (df["hour_local"] // 6).to_numpy()
    cols = [week, df[["is_holiday", "day_before_holiday", "day_after_holiday"]].to_numpy(dtype=float)]
    for b in range(4):
        m = block == b
        cols += [(hdd * m)[:, None], (cdd * m)[:, None]]
    return np.hstack(cols)


def fit_predict_regression(train, test, variant):
    """Least squares per BA on scaled demand. Rows with a missing input get a missing forecast."""
    scale = training_scale(train)
    pred = np.full(len(test), np.nan)
    for ba, tr in train.groupby("ba"):
        te_mask = (test["ba"] == ba).to_numpy()
        if not te_mask.any():
            continue
        tr = tr.dropna(subset=["y"])
        x = _design(tr, variant)
        y = tr["y"].to_numpy(dtype=float) / scale[ba]
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        beta = np.linalg.lstsq(x[ok], y[ok], rcond=None)[0]
        pred[te_mask] = (_design(test[te_mask], variant) @ beta) * scale[ba]
    return pred
