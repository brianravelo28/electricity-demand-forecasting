"""Download recent airport observations from the Iowa Environmental Mesonet (IEM).

NOAA's ISD archive has no observations after 2025-08-27, so this fills the period from
2025-08-20 to today for the same stations (the weather builder uses ISD wherever it has data). Output matches ingest_weather.py (temp_c,
dew_c, wind_ms) in data/raw/weather/obs_iem/, one parquet file per station.

IEM carries no quality flags. Readings that are physically impossible are set to missing:
temperature outside -50 to 55 C, a dew point above the temperature by more than 1 C or
outside -60 to 40 C, and wind speed outside 0 to 80 m/s.

    python scripts/ingest_iem.py
"""
import io
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "weather" / "obs_iem"
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
START = date(2025, 8, 20)  # before the earliest station's last ISD observation (2025-08-25), so none is left with a gap
KNOT = 0.514444  # metres per second
PAUSE = 0.5


def parse(text):
    df = pd.read_csv(io.StringIO(text), na_values=["M"])
    out = pd.DataFrame({"time": pd.to_datetime(df["valid"], utc=True), "report_type": "IEM-routine"})
    out["temp_c"] = ((df["tmpf"] - 32) * 5 / 9).round(1)
    out["temp_q"] = ""
    out["dew_c"] = ((df["dwpf"] - 32) * 5 / 9).round(1)
    out["dew_q"] = ""
    out["wind_ms"] = (df["sknt"] * KNOT).round(1)

    bad_temp = out["temp_c"].notna() & ~out["temp_c"].between(-50, 55)
    bad_dew = out["dew_c"].notna() & (~out["dew_c"].between(-60, 40) | (out["dew_c"] > out["temp_c"] + 1))
    bad_wind = out["wind_ms"].notna() & ~out["wind_ms"].between(0, 80)
    flagged = int(bad_temp.sum() + bad_dew.sum() + bad_wind.sum())
    out.loc[bad_temp, "temp_c"] = float("nan")
    out.loc[bad_dew, "dew_c"] = float("nan")
    out.loc[bad_wind, "wind_ms"] = float("nan")
    return out.sort_values("time").reset_index(drop=True), flagged


def fetch(code, end, tries=4):
    params = {
        "station": code, "data": ["tmpf", "dwpf", "sknt"],
        "year1": START.year, "month1": START.month, "day1": START.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no", "missing": "M",
        "trace": "T", "direct": "no", "report_type": 3,
    }
    for i in range(tries):
        try:
            resp = requests.get(URL, params=params, timeout=180)
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(10 * 2 ** i)
            continue
        if resp.status_code == 200:
            return resp.text
        if i == tries - 1:
            resp.raise_for_status()
        time.sleep(10 * 2 ** i)


def main():
    links = pd.read_parquet(ROOT / "data" / "weather_stations.parquet").drop_duplicates("station")
    OUT.mkdir(parents=True, exist_ok=True)
    end = date.today() + timedelta(days=1)
    for n, (_, s) in enumerate(links.iterrows(), 1):
        df, flagged = parse(fetch(s["icao"][1:], end))
        df.to_parquet(OUT / f"{s['station']}.parquet", index=False)
        last = df["time"].max() if len(df) else None
        print(f"{n}/{len(links)} {s['icao']}: {len(df)} obs, last {last}, impossible values set missing: {flagged}", flush=True)
        time.sleep(PAUSE)


if __name__ == "__main__":
    main()
