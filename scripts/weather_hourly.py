"""Build one hourly weather series per BA on the same clock as the EIA demand data.

Stamps are UTC and hour-ending, like EIA's `period`: the value stamped T is the latest good
observation with a time in (T minus 1 hour, T], so it never uses anything after T. Each
station uses NOAA ISD observations where they exist and the Iowa Mesonet after that. A BA is
the equal-weight mean over its stations that have a value that hour; `n_temp` counts them.
Missing hours stay missing. Reads data/raw/weather/ and data/weather_stations.parquet; writes
data/weather_hourly.parquet.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "data" / "raw" / "weather" / "obs"
OBS_IEM = ROOT / "data" / "raw" / "weather" / "obs_iem"
OUT = ROOT / "data" / "weather_hourly.parquet"
START = pd.Timestamp("2019-01-01", tz="UTC")
VARS = ["temp_c", "dew_c", "wind_ms"]


def station_hourly(station):
    isd = pd.concat([pd.read_parquet(f, columns=["time"] + VARS) for f in sorted(OBS.glob(f"{station}_*.parquet"))])
    iem = pd.read_parquet(OBS_IEM / f"{station}.parquet", columns=["time"] + VARS)
    iem = iem[iem["time"] > isd["time"].max()]
    obs = pd.concat([isd, iem]).sort_values("time")
    obs["stamp"] = obs["time"].dt.ceil("h")
    return obs.groupby("stamp")[VARS].last()  # last non-missing value per column


def main():
    links = pd.read_parquet(ROOT / "data" / "weather_stations.parquet")
    hourly = {st: station_hourly(st) for st in sorted(links["station"].unique())}
    end = min(h.index.max() for h in hourly.values()) - pd.Timedelta(hours=1)  # last fully observed stamp
    grid = pd.date_range(START, end, freq="h", name="period")

    frames = []
    for ba, g in links.groupby("ba", sort=False):
        per_station = [hourly[s].reindex(grid) for s in g["station"].unique()]
        stacked = pd.concat(per_station, keys=range(len(per_station)))
        by_hour = stacked.groupby(level=1)
        out = by_hour[VARS].mean()
        out["n_temp"] = by_hour["temp_c"].count()
        out.insert(0, "ba", ba)
        frames.append(out.reindex(grid).reset_index())
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT, index=False)

    pd.set_option("display.width", 200)
    print(f"{len(df)} rows: {df['ba'].nunique()} BAs x {len(grid)} hours, {grid[0]} to {grid[-1]}")
    s = df.groupby("ba", sort=False).agg(temp_covered=("temp_c", lambda x: x.notna().mean()),
                                         mean_stations=("n_temp", "mean"))
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()
