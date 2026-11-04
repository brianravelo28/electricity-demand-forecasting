"""Walk-forward evaluation and scoring (see docs/PLANNING.md).

Test blocks are calendar quarters from the start date. Each block is predicted by models fit
only on origin dates at least two days before the block's first origin date.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from electricity import models
from electricity.cleaning import clean_demand
from electricity.features import DAY, build_rows, split_by_origin

ROOT = Path(__file__).resolve().parent.parent
DF_EXCLUDED = {"FPC", "PSCO", "LGEE"}  # EIA's forecast is far off for these (see docs/PLANNING.md)
NAIVE = {"seasonal_naive": "base_seasonal_naive", "previous_day_naive": "lag_day", "same_day_avg": "base_same_day_avg"}
KEEP = ["ba", "origin_date", "local_date", "stamp", "hour_local", "is_holiday", "day_before_holiday",
        "day_after_holiday", "y", "eia_df"]
MODEL_ORDER = ["seasonal_naive", "previous_day_naive", "same_day_avg", "regression", "lightgbm", "eia_df"]
REGIMES = {
    "Winter Storm Uri (2021-02-13 to 02-19)": lambda d: d["local_date"].between("2021-02-13", "2021-02-19"),
    "Winter Storm Elliott (2022-12-21 to 12-27)": lambda d: d["local_date"].between("2022-12-21", "2022-12-27"),
    "Texas heat, ERCOT (2023-06-20 to 09-15)": lambda d: (d["ba"] == "ERCO") & d["local_date"].between("2023-06-20", "2023-09-15"),
    "Holidays and adjacent days": lambda d: d["is_holiday"] | d["day_before_holiday"] | d["day_after_holiday"],
}


def load_rows(variant, bas, tz_map):
    weather = pd.read_parquet(ROOT / "data" / "weather_hourly.parquet")
    frames = []
    for ba in bas:
        demand = pd.read_parquet(ROOT / "data" / "raw" / "hourly" / f"{ba}_D.parquet").set_index("period")["value"]
        demand, _ = clean_demand(demand.sort_index())
        wx = weather[weather["ba"] == ba].set_index("period")[["temp_c", "dew_c", "wind_ms"]]
        rows = build_rows(demand, tz_map[ba], wx, variant, ba=ba)
        forecast = pd.read_parquet(ROOT / "data" / "raw" / "hourly" / f"{ba}_DF.parquet").set_index("period")["value"]
        rows["eia_df"] = np.nan if ba in DF_EXCLUDED else rows["stamp"].map(forecast)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def walk_forward(rows, bas, variant, start="2021-01-01", rounds=400, max_blocks=None, log=print):
    last = rows["origin_date"].max()
    starts = pd.date_range(start, last, freq="QS")
    out = []
    for i, s in enumerate(starts):
        if max_blocks is not None and i >= max_blocks:
            break
        end = starts[i + 1] - DAY if i + 1 < len(starts) else last
        train, test = split_by_origin(rows, s, end)
        test = test.copy()
        test["pred_regression"] = models.fit_predict_regression(train, test, variant)
        test["pred_lightgbm"], _ = models.fit_predict_lgbm(train, test, variant, bas, rounds)
        for name, col in NAIVE.items():
            test[f"pred_{name}"] = test[col]
        test["pred_eia_df"] = test["eia_df"]
        test["is_peak"] = test["y"] == test.groupby(["ba", "local_date"])["y"].transform("max")
        out.append(test[KEEP + ["is_peak"] + [c for c in test.columns if c.startswith("pred_")]])
        log(f"block {s.date()} to {end.date()}: train {len(train):,} rows, test {len(test):,} rows")
    return pd.concat(out, ignore_index=True)


def common_sample(preds, with_df=False):
    cols = ["y"] + [f"pred_{m}" for m in MODEL_ORDER if m != "eia_df" or with_df]
    return preds.dropna(subset=cols)


def score(df, models_=None):
    models_ = models_ or [m for m in MODEL_ORDER if df[f"pred_{m}"].notna().any()]
    rows = []
    for m in models_:
        err = df[f"pred_{m}"] - df["y"]
        rel = err / df["y"]
        per_ba = (err ** 2).groupby(df["ba"]).mean().pow(0.5) / df.groupby("ba")["y"].mean()
        rows.append({"model": m, "mape_%": 100 * rel.abs().mean(), "nrmse_%": 100 * per_ba.mean(),
                     "bias_%": 100 * rel.mean(), "peak_hour_mape_%": 100 * rel[df["is_peak"]].abs().mean()})
    return pd.DataFrame(rows).set_index("model")


def mape_by(df, key, models_=None):
    models_ = models_ or [m for m in MODEL_ORDER if df[f"pred_{m}"].notna().any()]
    rel = {m: ((df[f"pred_{m}"] - df["y"]) / df["y"]).abs() * 100 for m in models_}
    keys = df[key] if isinstance(key, str) else key
    return pd.DataFrame(rel).groupby(keys).mean()


def regime_table(df):
    rows = {}
    models_ = [m for m in MODEL_ORDER if df[f"pred_{m}"].notna().any()]
    tagged = np.zeros(len(df), dtype=bool)
    for name, rule in REGIMES.items():
        mask = rule(df).to_numpy()
        tagged |= mask
        rows[name] = _regime_row(df[mask], models_)
    rows["Calm (none of the above)"] = _regime_row(df[~tagged], models_)
    return pd.DataFrame(rows).T


def _regime_row(df, models_):
    row = {"n": len(df)}
    for m in models_:
        row[m] = 100 * ((df[f"pred_{m}"] - df["y"]) / df["y"]).abs().mean() if len(df) else np.nan
    return row
