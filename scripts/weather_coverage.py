"""Coverage of hourly temperature per weather station and per BA, across the NOAA and Mesonet sources.

Reads data/raw/weather/obs/*.parquet (NOAA ISD), data/raw/weather/obs_iem/*.parquet (Iowa
Mesonet) and data/weather_stations.parquet; writes data/weather_coverage.csv. An hour counts
as covered when a station has at least one observation with a good temperature in it.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "data" / "raw" / "weather" / "obs"
OBS_IEM = ROOT / "data" / "raw" / "weather" / "obs_iem"
START = pd.Timestamp("2019-01-01", tz="UTC")
ISD_END = pd.Timestamp("2025-08-27", tz="UTC")  # last day in the ISD archive
LAST_ISD_YEAR = 2025


def main():
    links = pd.read_parquet(ROOT / "data" / "weather_stations.parquet")
    stations = sorted(links["station"].unique())

    hours, last, missing_years = {}, {}, []
    for st in stations:
        isd_files = sorted(OBS.glob(f"{st}_*.parquet"))
        years = {int(f.stem.split("_")[1]) for f in isd_files}
        missing_years += [(st, y) for y in range(2019, LAST_ISD_YEAR + 1) if y not in years]
        files = isd_files + sorted(OBS_IEM.glob(f"{st}.parquet"))
        df = pd.concat([pd.read_parquet(f, columns=["time", "temp_c"]) for f in files])
        good = df.dropna(subset=["temp_c"])
        hours[st] = pd.DatetimeIndex(good["time"].dt.floor("h").unique())
        last[st] = df["time"].max()

    end = min(last.values()).floor("h")
    grid = pd.date_range(START, end, freq="h")
    cov = pd.DataFrame({st: pd.Series(True, index=hours[st]).reindex(grid, fill_value=False) for st in stations})
    tail = cov.loc[cov.index > ISD_END]

    icao = links.drop_duplicates("station").set_index("station")["icao"]
    st_table = pd.DataFrame({"icao": icao, "hour_coverage": cov.mean(), "tail_coverage": tail.mean(),
                             "last_obs": pd.Series(last)}).sort_values("hour_coverage")
    st_table.to_csv(ROOT / "data" / "weather_coverage.csv")

    ba_rows = []
    for ba, g in links.groupby("ba", sort=False):
        cols = list(g["station"].unique())
        ba_rows.append({"ba": ba, "stations": len(cols), "any_all_years": cov[cols].any(axis=1).mean(),
                        "any_tail": tail[cols].any(axis=1).mean(), "all_tail": tail[cols].all(axis=1).mean()})
    ba_table = pd.DataFrame(ba_rows)

    pd.set_option("display.width", 200)
    print(f"stations: {len(stations)}   grid: {grid[0]} to {grid[-1]} ({len(grid)} hours; tail after {ISD_END.date()}: {len(tail)})")
    print("missing ISD station-years (2019 to 2025):", missing_years if missing_years else "none")
    print("\nStations with lowest hourly coverage:")
    print(st_table.head(8).round({"hour_coverage": 4, "tail_coverage": 4}).to_string())
    print("\nOldest last observation:", st_table["last_obs"].min())
    print("\nPer BA, share of hours with a good temperature (any station; tail = after the ISD archive ends):")
    print(ba_table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
