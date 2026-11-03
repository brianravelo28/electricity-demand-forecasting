"""Target and feature construction for next-day hourly demand (see docs/PLANNING.md).

Series sit on a regular UTC hour-ending grid: the value stamped T covers T minus 1 hour to T.
One row is one origin date d (local) and one target hour. The forecast is made at 10:00 local
on d, uses demand only through the stamp at 08:00 local and weather only through the stamp at
09:00 local, and targets the hours of local date d+1.
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

ORIGIN_HOUR = 10
LAST_DEMAND_HOUR = 8
LAST_WEATHER_HOUR = 9
WEEK = 168
HDD_BASE_C = 18.3  # 65 F
MIN_SHARE = 0.75  # share of a trailing window that must be present
GAP_DAYS = 2  # training origin dates stop this many days before the first test origin date
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)


def _check_utc(index):
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or str(index.tz) != "UTC":
        raise ValueError("index must be a UTC DatetimeIndex of hour-ending stamps")
    if not index.is_unique:
        raise ValueError("duplicate stamps in index")


def _regular(series):
    _check_utc(series.index)
    series = series.sort_index()
    return series.reindex(pd.date_range(series.index[0], series.index[-1], freq="h"))


def _positions(stamps, origin):
    hours = np.asarray((stamps - origin) / HOUR, dtype=float)
    return np.rint(hours).astype(np.int64)


def _at(values, pos):
    """values[pos], or NaN where pos falls outside the grid."""
    out = np.full(pos.shape, np.nan)
    ok = (pos >= 0) & (pos < len(values))
    out[ok] = values[pos[ok]]
    return out


def _local_utc(dates, tz, hour=0):
    """UTC instants of `hour`:00 local wall-clock time on each (naive) date."""
    return (dates + pd.Timedelta(hours=hour)).tz_localize(tz).tz_convert("UTC")


def _trailing(values, window, how):
    s = pd.Series(values).rolling(window, min_periods=int(np.ceil(MIN_SHARE * window)))
    return (s.mean() if how == "mean" else s.max()).to_numpy()


def build_rows(demand, tz, weather=None, variant=None, ba=None):
    """One row per origin date and target hour.

    demand:  Series on UTC hour-ending stamps (MWh per hour).
    tz:      IANA zone of the BA's load.
    weather: DataFrame with temp_c, dew_c, wind_ms on UTC hour-ending stamps, or None.
    variant: None (no weather), "A" (weather up to 09:00 local on the origin date) or
             "B" (A plus observed weather for the target day, an upper bound).
    """
    if variant not in (None, "A", "B"):
        raise ValueError("variant must be None, 'A' or 'B'")
    if variant is not None and weather is None:
        raise ValueError("weather is required for variants A and B")

    demand = _regular(demand)
    grid0 = demand.index[0]
    v = demand.to_numpy(dtype=float)
    if weather is not None:
        _check_utc(weather.index)
        weather = weather.reindex(demand.index)

    first = grid0.tz_convert(tz).tz_localize(None).normalize() + DAY
    last = demand.index[-1].tz_convert(tz).tz_localize(None).normalize() - DAY
    dates = pd.date_range(first, last, freq="D")
    m0 = _local_utc(dates + DAY, tz)
    m1 = _local_utc(dates + 2 * DAY, tz)
    keep = np.asarray(m1 <= demand.index[-1])
    dates, m0, m1 = dates[keep], m0[keep], m1[keep]
    origin = _local_utc(dates, tz, ORIGIN_HOUR)
    pos_ld = _positions(_local_utc(dates, tz, LAST_DEMAND_HOUR), grid0)
    pos_wd = _positions(_local_utc(dates, tz, LAST_WEATHER_HOUR), grid0)
    n_targets = np.asarray((m1 - m0) / HOUR, dtype=float).astype(int)  # 23, 24 or 25

    d_idx = np.repeat(np.arange(len(dates)), n_targets)
    k = np.concatenate([np.arange(1, n + 1) for n in n_targets])
    stamp = m0[d_idx] + pd.to_timedelta(k, unit="h")
    pos_t = _positions(stamp, grid0)
    pos_ld_row = pos_ld[d_idx]

    local = (stamp - HOUR).tz_convert(tz)  # the hour that stamp T closes, on the local clock
    local_date = local.tz_localize(None).normalize()
    holidays = USFederalHolidayCalendar().holidays(local_date.min() - 2 * DAY, local_date.max() + 2 * DAY)
    doy = np.asarray(local.day_of_year, dtype=float)

    out = pd.DataFrame({
        "origin_date": dates[d_idx],
        "local_date": local_date,
        "stamp": stamp,
        "hour_local": np.asarray(local.hour),
        "weekday": np.asarray(local.day_of_week),
        "month": np.asarray(local.month),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "is_holiday": np.asarray(local_date.isin(holidays)),
        "day_before_holiday": np.asarray((local_date + DAY).isin(holidays)),
        "day_after_holiday": np.asarray((local_date - DAY).isin(holidays)),
        "hours_in_day": n_targets[d_idx],
        "horizon": np.asarray(((stamp - HOUR) - origin[d_idx]) / HOUR, dtype=float).astype(int),
    })
    if ba is not None:
        out.insert(0, "ba", ba)

    # demand known at the origin: stamps up to the 08:00 local stamp
    out["demand_last"] = _at(v, pos_ld_row)
    out["demand_mean_24h"] = _at(_trailing(v, 24, "mean"), pos_ld_row)
    out["demand_max_24h"] = _at(_trailing(v, 24, "max"), pos_ld_row)
    out["demand_mean_168h"] = _at(_trailing(v, WEEK, "mean"), pos_ld_row)
    out["demand_change_3h"] = _at(v, pos_ld_row) - _at(v, pos_ld_row - 3)
    lags = [_at(v, pos_t - WEEK * w) for w in range(1, 5)]
    for w, lag in enumerate(lags, 1):
        out[f"lag_week_{w}"] = lag
    same_day_known = (pos_t - 24) <= pos_ld_row  # yesterday's hour is known only up to 08:00 local
    out["lag_day"] = np.where(same_day_known, _at(v, pos_t - 24), _at(v, pos_t - 48))
    out["lag_day_age"] = np.where(same_day_known, 1, 2)

    if variant is not None:
        temp = weather["temp_c"].to_numpy(dtype=float)
        dew = weather["dew_c"].to_numpy(dtype=float)
        wind = weather["wind_ms"].to_numpy(dtype=float)
        pos_wd_row = pos_wd[d_idx]
        out["temp_last"] = _at(temp, pos_wd_row)
        out["temp_mean_24h"] = _at(_trailing(temp, 24, "mean"), pos_wd_row)
        out["temp_change_3h"] = _at(temp, pos_wd_row) - _at(temp, pos_wd_row - 3)
        out["dew_last"] = _at(dew, pos_wd_row)
        out["wind_last"] = _at(wind, pos_wd_row)
    if variant == "B":
        t_target = _at(temp, pos_t)
        out["temp_target"] = t_target
        out["dew_target"] = _at(dew, pos_t)
        out["wind_target"] = _at(wind, pos_t)
        g = pd.Series(t_target).groupby(d_idx)
        out["temp_day_max"] = g.transform("max").to_numpy()
        out["temp_day_min"] = g.transform("min").to_numpy()
        out["temp_day_mean"] = g.transform("mean").to_numpy()
        out["hdd_day"] = pd.Series(np.maximum(0, HDD_BASE_C - t_target)).groupby(d_idx).transform("mean").to_numpy()
        out["cdd_day"] = pd.Series(np.maximum(0, t_target - HDD_BASE_C)).groupby(d_idx).transform("mean").to_numpy()

    out["base_seasonal_naive"] = lags[0]
    out["base_same_day_avg"] = np.mean(lags, axis=0)  # missing if any of the four weeks is missing
    out["y"] = _at(v, pos_t)
    return out


def split_by_origin(rows, test_start, test_end=None):
    """Train and test rows. Training stops GAP_DAYS before the first test origin date, because a
    row's targets run to local midnight of d+2."""
    test_start = pd.Timestamp(test_start)
    train = rows[rows["origin_date"] <= test_start - GAP_DAYS * DAY]
    test = rows[rows["origin_date"] >= test_start]
    if test_end is not None:
        test = test[test["origin_date"] <= pd.Timestamp(test_end)]
    return train, test
