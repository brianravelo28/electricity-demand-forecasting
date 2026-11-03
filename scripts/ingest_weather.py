"""Download raw hourly surface observations (NOAA ISD "global hourly") for the BA weather stations.

Source: NOAA's public S3 bucket noaa-global-hourly-pds (the old ncei.noaa.gov path now
redirects there). One parquet file per station and year in data/raw/weather/obs/, keeping
only the parsed temperature, dew point and wind speed. Existing files are skipped, except
the current year, which is refreshed.

    python scripts/ingest_weather.py
"""
import io
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "weather" / "obs"
URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{station}.csv"
FIRST_YEAR = 2019
PAUSE = 0.2

# ISD quality flags (isd-format-document): 2, 3, 6 and 7 mean suspect or erroneous.
# 9 means "passed gross limits if the element is present", so a real value is kept.
BAD_QUALITY = set("2367")
COLUMNS = ["DATE", "REPORT_TYPE", "TMP", "DEW", "WND"]


def parse_field(series, scale, missing):
    """Split an ISD 'value,quality[,...]' field into a float and its quality code."""
    parts = series.str.split(",", expand=True)
    value = pd.to_numeric(parts[0], errors="coerce")
    quality = parts[1].fillna("")
    value = value.where(value != missing)
    value = value.where(~quality.isin(BAD_QUALITY)) / scale
    return value, quality


def parse(raw):
    df = pd.read_csv(io.BytesIO(raw), dtype=str, usecols=COLUMNS, low_memory=False)
    out = pd.DataFrame({
        "time": pd.to_datetime(df["DATE"], utc=True),
        "report_type": df["REPORT_TYPE"].str.strip(),
    })
    out["temp_c"], out["temp_q"] = parse_field(df["TMP"], 10, 9999)
    out["dew_c"], out["dew_q"] = parse_field(df["DEW"], 10, 9999)
    wnd = df["WND"].str.split(",", expand=True)
    speed = pd.to_numeric(wnd[3], errors="coerce").where(lambda s: s != 9999)
    speed_q = wnd[4].fillna("")
    out["wind_ms"] = speed.where(~speed_q.isin(BAD_QUALITY)) / 10
    return out.sort_values("time").reset_index(drop=True)


def fetch(url, tries=4):
    for i in range(tries):
        try:
            resp = requests.get(url, timeout=120)
        except (requests.ConnectionError, requests.Timeout):
            if i == tries - 1:
                raise
            time.sleep(10 * 2 ** i)
            continue
        if resp.status_code == 200:
            return resp.content
        if resp.status_code == 404:
            return None
        if i == tries - 1:
            resp.raise_for_status()
        time.sleep(10 * 2 ** i)


def main():
    stations = sorted(pd.read_parquet(ROOT / "data" / "weather_stations.parquet")["station"].unique())
    years = range(FIRST_YEAR, date.today().year + 1)
    OUT.mkdir(parents=True, exist_ok=True)
    total = len(stations) * len(years)
    n = 0
    for station in stations:
        for year in years:
            n += 1
            path = OUT / f"{station}_{year}.parquet"
            if path.exists() and year != date.today().year:
                continue
            raw = fetch(URL.format(year=year, station=station))
            if raw is None:
                print(f"{n}/{total} {station} {year}: no file", flush=True)
                continue
            df = parse(raw)
            df.to_parquet(path, index=False)
            print(f"{n}/{total} {station} {year}: {len(df)} obs", flush=True)
            time.sleep(PAUSE)


if __name__ == "__main__":
    main()
