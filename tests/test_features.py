"""Leakage and calendar tests for electricity.features on synthetic series.

Demand and weather values equal their own position on the hourly grid, so a feature's value
says which stamp it read: any feature larger than its cutoff position read the future.
"""
import numpy as np
import pandas as pd
import pytest

from electricity.features import GAP_DAYS, build_rows, split_by_origin

ZONES = ["America/Chicago", "America/New_York", "America/Phoenix", "America/Los_Angeles"]
IDX = pd.date_range("2024-01-01", "2025-01-20", freq="h", tz="UTC")
POS = np.arange(len(IDX), dtype=float)
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)
DEMAND_FEATURES = ["demand_last", "demand_mean_24h", "demand_max_24h", "demand_mean_168h",
                   "lag_week_1", "lag_week_2", "lag_week_3", "lag_week_4", "lag_day"]


def pos_of(ts):
    return int((ts - IDX[0]) / HOUR)


def local_stamp(date, hour, tz):
    """The UTC instant of `hour`:00 local wall-clock time on a date, computed independently."""
    return pd.Timestamp(f"{pd.Timestamp(date):%Y-%m-%d} {hour:02d}:00").tz_localize(tz).tz_convert("UTC")


def expected_pos(rows, hour, tz):
    per_date = {d: pos_of(local_stamp(d, hour, tz)) for d in pd.unique(rows["origin_date"])}
    return rows["origin_date"].map(per_date).to_numpy()


def stamp_pos(rows):
    return ((rows["stamp"] - IDX[0]) / HOUR).to_numpy()


def make_inputs():
    demand = pd.Series(POS.copy(), index=IDX)
    weather = pd.DataFrame({"temp_c": POS.copy(), "dew_c": POS.copy(), "wind_ms": POS.copy()}, index=IDX)
    return demand, weather


@pytest.fixture(scope="module", params=ZONES)
def built(request):
    tz = request.param
    demand, weather = make_inputs()
    return tz, build_rows(demand, tz, weather, "B")


def test_demand_inputs_stop_at_0800_local(built):
    tz, rows = built
    p_ld = expected_pos(rows, 8, tz)
    assert (rows[DEMAND_FEATURES].max(axis=1) <= p_ld).all()
    assert (rows["demand_last"] == p_ld).all()
    assert (rows["demand_change_3h"] == 3).all()
    assert (stamp_pos(rows) > p_ld).all()  # every target is after the last input
    assert (rows["y"] == stamp_pos(rows)).all()


def test_weather_inputs_stop_at_0900_local(built):
    tz, rows = built
    p_wd = expected_pos(rows, 9, tz)
    for col in ["temp_last", "dew_last", "wind_last"]:
        assert (rows[col] == p_wd).all()
    mean = rows["temp_mean_24h"]
    assert mean.notna().mean() > 0.99  # missing only where the 24-hour window starts before the grid
    assert (mean.dropna() <= p_wd[mean.notna().to_numpy()]).all()
    assert (rows["temp_change_3h"] == 3).all()


def test_variant_b_reads_the_target_day(built):
    tz, rows = built
    assert (rows["temp_target"] == stamp_pos(rows)).all()
    last = rows["origin_date"].map({d: pos_of(local_stamp(pd.Timestamp(d) + 2 * DAY, 0, tz)) for d in pd.unique(rows["origin_date"])})
    first = rows["origin_date"].map({d: pos_of(local_stamp(pd.Timestamp(d) + DAY, 0, tz)) + 1 for d in pd.unique(rows["origin_date"])})
    assert (rows["temp_day_max"] == last).all()
    assert (rows["temp_day_min"] == first).all()


def test_variants_carry_only_their_columns():
    demand, weather = make_inputs()
    none_cols = set(build_rows(demand, "America/Chicago").columns)
    a_cols = set(build_rows(demand, "America/Chicago", weather, "A").columns)
    b_cols = set(build_rows(demand, "America/Chicago", weather, "B").columns)
    assert "temp_last" not in none_cols
    assert "temp_last" in a_cols and "temp_target" not in a_cols
    assert {"temp_target", "temp_day_max", "hdd_day"} <= b_cols
    with pytest.raises(ValueError):
        build_rows(demand, "America/Chicago", None, "A")


def test_perturbing_the_future_leaves_features_unchanged(built):
    tz, _ = built
    demand, weather = make_inputs()
    d = pd.Timestamp("2024-06-18")
    base = build_rows(demand, tz, weather, "A")
    demand2, weather2 = demand.copy(), weather.copy()
    demand2[demand2.index > local_stamp(d, 8, tz)] = 1e9
    weather2.loc[weather2.index > local_stamp(d, 9, tz)] = 1e9
    changed = build_rows(demand2, tz, weather2, "A")
    a = base[base["origin_date"] == d].drop(columns="y").reset_index(drop=True)
    b = changed[changed["origin_date"] == d].drop(columns="y").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_the_perturbation_test_can_fail(built):
    tz, _ = built
    demand, weather = make_inputs()
    d = pd.Timestamp("2024-06-18")
    demand2 = demand.copy()
    demand2[local_stamp(d, 8, tz)] = 1e9  # the last allowed stamp, so features must react
    base = build_rows(demand, tz, weather, "A")
    changed = build_rows(demand2, tz, weather, "A")
    assert base.loc[base["origin_date"] == d, "demand_last"].iloc[0] != changed.loc[changed["origin_date"] == d, "demand_last"].iloc[0]


def test_missing_values_are_not_filled(built):
    tz, _ = built
    demand, weather = make_inputs()
    hole = IDX[5000]
    demand2 = demand.copy()
    demand2[hole] = np.nan
    rows = build_rows(demand2, tz, weather, "A")
    row = rows[rows["stamp"] == hole + pd.Timedelta(hours=168)]
    assert len(row) == 1
    assert row["lag_week_1"].isna().all() and row["base_seasonal_naive"].isna().all()
    assert row["base_same_day_avg"].isna().all()
    assert rows.loc[rows["stamp"] == hole, "y"].isna().all()


def test_target_hours_per_day_follow_the_local_clock(built):
    tz, rows = built
    count = rows.groupby("origin_date").size()
    spring, fall = (23, 25) if tz != "America/Phoenix" else (24, 24)
    assert count[pd.Timestamp("2024-03-09")] == spring  # target day 2024-03-10
    assert count[pd.Timestamp("2024-11-02")] == fall    # target day 2024-11-03
    assert count[pd.Timestamp("2024-06-18")] == 24
    assert (rows.groupby("origin_date")["hours_in_day"].first() == count).all()


def test_horizon_range(built):
    tz, rows = built
    normal = rows[rows["hours_in_day"] == 24]
    assert normal["horizon"].min() == 14 and normal["horizon"].max() == 37
    if tz != "America/Phoenix":
        spring = rows[rows["origin_date"] == pd.Timestamp("2024-03-09")]
        fall = rows[rows["origin_date"] == pd.Timestamp("2024-11-02")]
        assert spring["horizon"].max() == 36 and fall["horizon"].max() == 38


def test_yesterday_is_used_only_for_hours_known_at_the_origin(built):
    tz, rows = built
    n = rows[rows["hours_in_day"] == 24]
    assert ((n["lag_day_age"] == 1) == (n["hour_local"] <= 7)).all()
    expected = stamp_pos(n) - np.where(n["lag_day_age"] == 1, 24, 48)
    inside = expected >= 0  # a lag that reaches before the start of the grid is missing, not filled
    assert (n["lag_day"].to_numpy()[inside] == expected[inside]).all()
    assert n["lag_day"].isna().to_numpy()[~inside].all()


def test_calendar_uses_the_local_clock(built):
    tz, rows = built
    local = (rows["stamp"] - HOUR).dt.tz_convert(tz)
    assert (rows["hour_local"] == local.dt.hour).all()
    assert (rows["local_date"] == local.dt.tz_localize(None).dt.normalize()).all()
    assert (rows["hour_local"] != (rows["stamp"] - HOUR).dt.hour).any()  # not the UTC hour for any zone but a UTC one


def test_holiday_flags(built):
    _, rows = built
    day = lambda s: rows[rows["local_date"] == pd.Timestamp(s)]
    assert day("2024-07-04")["is_holiday"].all()
    assert day("2024-07-03")["day_before_holiday"].all() and not day("2024-07-03")["is_holiday"].any()
    assert day("2024-07-05")["day_after_holiday"].all()
    plain = day("2024-07-10")
    assert not (plain["is_holiday"] | plain["day_before_holiday"] | plain["day_after_holiday"]).any()


def test_training_stops_two_days_before_the_first_test_origin(built):
    tz, rows = built
    t0 = pd.Timestamp("2024-06-03")
    train, test = split_by_origin(rows, t0, "2024-06-10")
    origin = local_stamp(t0, 10, tz)
    assert train["stamp"].max() <= origin
    assert train["origin_date"].max() == t0 - GAP_DAYS * DAY
    assert test["origin_date"].min() == t0 and test["origin_date"].max() == pd.Timestamp("2024-06-10")
    # one day closer and a training target would fall after the test origin
    assert rows.loc[rows["origin_date"] == t0 - DAY, "stamp"].max() > origin


def test_inputs_must_be_utc():
    demand, _ = make_inputs()
    with pytest.raises(ValueError):
        build_rows(demand.tz_convert("America/Chicago"), "America/Chicago")
    with pytest.raises(ValueError):
        build_rows(demand.tz_localize(None), "America/Chicago")
