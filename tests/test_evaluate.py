"""Tests for the scoring and walk-forward code in electricity.evaluate, on small synthetic frames."""
import re

import numpy as np
import pandas as pd
import pytest

from electricity import evaluate
from electricity.features import build_rows

ALL_MODELS = evaluate.MODEL_ORDER


def frame(rows):
    """Rows of (ba, local_date, y, pred, is_peak) with every model column set to the same forecast."""
    df = pd.DataFrame(rows, columns=["ba", "local_date", "y", "pred", "is_peak"])
    df["local_date"] = pd.to_datetime(df["local_date"])
    for m in ALL_MODELS:
        df[f"pred_{m}"] = df["pred"]
    df["is_holiday"] = False
    df["day_before_holiday"] = False
    df["day_after_holiday"] = False
    return df


def test_common_sample_drops_rows_missing_any_model_or_the_target():
    df = frame([("A", "2024-01-01", 100.0, 110.0, True)] * 4)
    df.loc[1, "pred_regression"] = np.nan
    df.loc[2, "y"] = np.nan
    df.loc[3, "pred_eia_df"] = np.nan  # only matters when EIA's forecast is required
    assert len(evaluate.common_sample(df)) == 2
    assert len(evaluate.common_sample(df, with_df=True)) == 1


def test_score_matches_hand_computed_figures():
    df = frame([
        ("A", "2024-01-01", 100.0, 110.0, False),
        ("A", "2024-01-01", 200.0, 180.0, True),
        ("B", "2024-01-01", 50.0, 50.0, False),
        ("B", "2024-01-01", 50.0, 60.0, True),
    ])
    out = evaluate.score(df, ["lightgbm"]).loc["lightgbm"]
    assert out["mape_%"] == pytest.approx(100 * np.mean([0.10, 0.10, 0.0, 0.20]))
    assert out["bias_%"] == pytest.approx(100 * np.mean([0.10, -0.10, 0.0, 0.20]))  # above 0 means too high
    assert out["peak_hour_mape_%"] == pytest.approx(100 * np.mean([0.10, 0.20]))
    rmse_a, rmse_b = np.sqrt((10 ** 2 + 20 ** 2) / 2), np.sqrt((0 ** 2 + 10 ** 2) / 2)
    assert out["nrmse_%"] == pytest.approx(100 * np.mean([rmse_a / 150, rmse_b / 50]))


def test_a_perfect_forecast_scores_zero():
    df = frame([("A", "2024-01-01", 100.0, 100.0, True), ("A", "2024-01-02", 120.0, 120.0, True)])
    out = evaluate.score(df, ["lightgbm"]).loc["lightgbm"]
    assert (out == 0).all()


def test_mape_by_groups_by_a_column_or_a_series():
    df = frame([
        ("A", "2023-06-01", 100.0, 110.0, True),
        ("A", "2024-06-01", 100.0, 130.0, True),
        ("B", "2024-06-01", 100.0, 100.0, True),
    ])
    by_ba = evaluate.mape_by(df, "ba", ["lightgbm"])
    assert by_ba.loc["A", "lightgbm"] == pytest.approx(20.0) and by_ba.loc["B", "lightgbm"] == 0
    by_year = evaluate.mape_by(df, df["local_date"].dt.year, ["lightgbm"])
    assert by_year.loc[2023, "lightgbm"] == pytest.approx(10.0)
    assert by_year.loc[2024, "lightgbm"] == pytest.approx(15.0)


def test_regimes_select_the_documented_dates_and_the_calm_row_is_the_rest():
    df = frame([
        ("PJM", "2021-02-15", 100.0, 110.0, True),   # Uri
        ("PJM", "2021-02-20", 100.0, 110.0, True),   # the day after Uri ends: calm
        ("PJM", "2022-12-24", 100.0, 120.0, True),   # Elliott
        ("ERCO", "2023-07-04", 100.0, 130.0, True),  # Texas heat
        ("PJM", "2023-07-04", 100.0, 130.0, True),   # same date elsewhere: not the Texas heat
        ("PJM", "2024-03-05", 100.0, 100.0, True),   # calm
    ])
    df.loc[4, "is_holiday"] = True  # the PJM row on 2023-07-04
    df.loc[3, "is_holiday"] = True  # and the ERCO row
    table = evaluate.regime_table(df)
    n = table["n"]
    assert n["Winter Storm Uri (2021-02-13 to 02-19)"] == 1
    assert n["Winter Storm Elliott (2022-12-21 to 12-27)"] == 1
    assert n["Texas heat, ERCOT (2023-06-20 to 09-15)"] == 1
    assert n["Holidays and adjacent days"] == 2
    assert n["Calm (none of the above)"] == 2  # 2021-02-20 and 2024-03-05


def test_regimes_overlap_so_a_holiday_inside_a_storm_counts_in_both():
    df = frame([("PJM", "2021-02-15", 100.0, 110.0, True)])  # Presidents' Day inside Uri
    df["is_holiday"] = True
    table = evaluate.regime_table(df)
    assert table.loc["Winter Storm Uri (2021-02-13 to 02-19)", "n"] == 1
    assert table.loc["Holidays and adjacent days", "n"] == 1
    assert table.loc["Calm (none of the above)", "n"] == 0


def test_an_empty_regime_reports_nan_not_a_crash():
    df = frame([("PJM", "2024-03-05", 100.0, 100.0, True)])
    table = evaluate.regime_table(df)
    assert table.loc["Winter Storm Uri (2021-02-13 to 02-19)", "n"] == 0
    assert np.isnan(table.loc["Winter Storm Uri (2021-02-13 to 02-19)", "lightgbm"])


@pytest.fixture(scope="module")
def synthetic_rows():
    idx = pd.date_range("2019-01-01", "2021-04-20", freq="h", tz="UTC")
    hours = np.arange(len(idx))
    rng = np.random.default_rng(0)
    demand = pd.Series(1000 + 200 * np.sin(2 * np.pi * hours / 24) + 80 * np.sin(2 * np.pi * hours / 168)
                       + 60 * np.sin(2 * np.pi * hours / 8766) + rng.normal(0, 5, len(idx)), index=idx)
    temp = 15 + 12 * np.sin(2 * np.pi * hours / 8766) + 4 * np.sin(2 * np.pi * hours / 24)
    weather = pd.DataFrame({"temp_c": temp, "dew_c": temp - 5, "wind_ms": 3.0}, index=idx)
    rows = build_rows(demand, "America/Chicago", weather, "A", ba="X")
    rows["eia_df"] = rows["y"] * 1.02
    return rows


def test_walk_forward_trains_only_on_earlier_origins_and_tests_only_the_block(synthetic_rows):
    logs = []
    out = evaluate.walk_forward(synthetic_rows, ["X"], "A", start="2021-01-01", rounds=5, max_blocks=1, log=logs.append)
    assert out["origin_date"].min() >= pd.Timestamp("2021-01-01")
    assert out["origin_date"].max() <= pd.Timestamp("2021-03-31")
    trained = int(re.search(r"train ([\d,]+) rows", logs[0]).group(1).replace(",", ""))
    assert trained == int((synthetic_rows["origin_date"] <= pd.Timestamp("2020-12-30")).sum())  # two days before the block


def test_walk_forward_output_has_every_model_and_matching_baselines(synthetic_rows):
    out = evaluate.walk_forward(synthetic_rows, ["X"], "A", start="2021-01-01", rounds=100, max_blocks=1, log=lambda s: None)
    for m in ALL_MODELS:
        assert f"pred_{m}" in out
    keyed = out.merge(synthetic_rows[["stamp", "lag_day", "base_seasonal_naive"]], on="stamp")
    assert np.allclose(keyed["pred_previous_day_naive"], keyed["lag_day"], equal_nan=True)
    assert np.allclose(keyed["pred_seasonal_naive"], keyed["base_seasonal_naive"], equal_nan=True)
    assert np.allclose(out["pred_eia_df"], out["eia_df"], equal_nan=True)
    assert np.isfinite(out["pred_lightgbm"]).mean() > 0.99
    # a smooth synthetic series is easy: the model should be close to it
    assert evaluate.score(evaluate.common_sample(out), ["lightgbm"]).loc["lightgbm", "mape_%"] < 10


def test_each_day_marks_its_peak_hour(synthetic_rows):
    out = evaluate.walk_forward(synthetic_rows, ["X"], "A", start="2021-01-01", rounds=3, max_blocks=1, log=lambda s: None)
    per_day = out.groupby(["ba", "local_date"])["is_peak"].sum()
    assert (per_day >= 1).all()
    peak = out[out["is_peak"]]
    assert (peak["y"] == out.groupby(["ba", "local_date"])["y"].transform("max")[peak.index]).all()
