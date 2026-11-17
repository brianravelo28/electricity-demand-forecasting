"""Regression and LightGBM forecasts for the walk-forward evaluation (see docs/METHODOLOGY.md).

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
    """N: no weather; A: weather observed up to the origin; B: A plus the target day's observed weather."""
    weather = {"N": [], "A": WEATHER_A, "B": WEATHER_A + WEATHER_B}[variant]
    return LEVEL_COLS + CALENDAR_COLS + weather


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


RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
RIDGE_VALIDATION_DAYS = 90
RIDGE_CHUNK = 200_000
RIDGE_DEGREE_TEMPS = {"N": [], "A": ["temp_last", "temp_mean_24h"], "B": ["temp_last", "temp_mean_24h", "temp_target", "temp_day_mean"]}


def _ridge_parts(df, scale, bas, variant):
    """The ridge design before standardizing: a numeric block and a one-hot block.

    Demand levels are divided by each BA's training mean, as for LightGBM. Weather enters as
    given and as heating and cooling degrees, the latter also split by six-hour block of the day.
    """
    s = df["ba"].map(scale).to_numpy(dtype=float)
    num = {c: df[c].to_numpy(dtype=float) / s for c in LEVEL_COLS}
    for c in ["horizon", "lag_day_age", "hours_in_day", "doy_sin", "doy_cos",
              "is_holiday", "day_before_holiday", "day_after_holiday"]:
        num[c] = df[c].to_numpy(dtype=float)
    for c in feature_columns(variant):
        if c in WEATHER_A or c in WEATHER_B:
            num[c] = df[c].to_numpy(dtype=float)
    hour = df["hour_local"].to_numpy(dtype=int)
    for t in RIDGE_DEGREE_TEMPS[variant]:
        temp = df[t].to_numpy(dtype=float)
        for name, degrees in (("hdd", np.maximum(0.0, HDD_BASE_C - temp)), ("cdd", np.maximum(0.0, temp - HDD_BASE_C))):
            num[f"{name}_{t}"] = degrees
            for b in range(4):
                num[f"{name}_{t}_b{b}"] = degrees * ((hour // 6) == b)
    n = len(df)
    blocks = []
    for values, size in ((hour, 24), (df["weekday"].to_numpy(dtype=int), 7), (df["month"].to_numpy(dtype=int) - 1, 12)):
        m = np.zeros((n, size))
        m[np.arange(n), values] = 1.0
        blocks.append(m)
    codes = pd.Categorical(df["ba"], categories=list(bas)).codes
    m = np.zeros((n, len(bas)))
    known = codes >= 0
    m[np.arange(n)[known], codes[known]] = 1.0
    blocks.append(m)
    return np.column_stack(list(num.values())), np.hstack(blocks)


def _ridge_design(df, scale, bas, variant, mu, sd):
    numeric, one_hot = _ridge_parts(df, scale, bas, variant)
    return np.hstack([np.ones((len(df), 1)), (numeric - mu) / sd, one_hot])


def fit_predict_ridge(train, test, variant, bas, alphas=RIDGE_ALPHAS):
    """Ridge regression on the same inputs as LightGBM (a linear model with hour, weekday, month and BA effects).

    The penalty is chosen from `alphas` on the last 90 days of the training window, fitting only on rows
    whose targets end before that stretch starts, so nothing from the test block is used. The intercept
    is not penalized. Rows with a missing input get a missing forecast.
    """
    scale = training_scale(train)
    train = train.dropna(subset=["y"])
    sample = train.sample(min(len(train), 200_000), random_state=0)
    numeric, _ = _ridge_parts(sample, scale, bas, variant)
    finite = np.isfinite(numeric).all(axis=1)
    mu, sd = numeric[finite].mean(axis=0), numeric[finite].std(axis=0)
    sd[sd == 0] = 1.0

    cut = train["origin_date"].max() - pd.Timedelta(days=RIDGE_VALIDATION_DAYS)
    fit_end = cut - pd.Timedelta(days=2)
    gram, xty, xv, yv = {}, {}, [], []
    for start in range(0, len(train), RIDGE_CHUNK):
        chunk = train.iloc[start:start + RIDGE_CHUNK]
        x = _ridge_design(chunk, scale, bas, variant, mu, sd)
        y = (chunk["y"] / chunk["ba"].map(scale)).to_numpy(dtype=float)
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        origin = chunk["origin_date"]
        groups = {"fit": (origin <= fit_end).to_numpy()[ok], "val": (origin > cut).to_numpy()[ok]}
        groups["gap"] = ~groups["fit"] & ~groups["val"]
        x, y = x[ok], y[ok]
        for name, mask in groups.items():
            gram[name] = gram.get(name, 0) + x[mask].T @ x[mask]
            xty[name] = xty.get(name, 0) + x[mask].T @ y[mask]
        xv.append(x[groups["val"]])
        yv.append(y[groups["val"]])
    xv, yv = np.vstack(xv), np.concatenate(yv)
    penalty = np.eye(gram["fit"].shape[0])
    penalty[0, 0] = 0.0  # the intercept

    def solve(g, b, alpha):
        return np.linalg.solve(g + alpha * penalty, b)

    if len(yv) and len(xv):
        errors = {a: np.mean(np.abs(xv @ solve(gram["fit"], xty["fit"], a) - yv) / yv) for a in alphas}
        alpha = min(errors, key=errors.get)
    else:
        alpha = 1.0
    beta = solve(sum(gram.values()), sum(xty.values()), alpha)

    pred = np.full(len(test), np.nan)
    for start in range(0, len(test), RIDGE_CHUNK):
        chunk = test.iloc[start:start + RIDGE_CHUNK]
        pred[start:start + len(chunk)] = (_ridge_design(chunk, scale, bas, variant, mu, sd) @ beta) \
            * chunk["ba"].map(scale).to_numpy(dtype=float)
    return pred


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
