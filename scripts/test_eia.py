"""Check EIA API access: key validity, history depth, pagination and timestamp behaviour.

Reads EIA_API_KEY from the environment (or a local .env). The key travels in the
query string, so errors are scrubbed before they are shown.
"""
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

BASE = "https://api.eia.gov/v2/electricity/rto/region-data"
PAGE = 5000
BAS = ["ERCO", "PJM", "CISO"]
PAUSE = 0.5


def get(route, params):
    key = os.environ["EIA_API_KEY"]
    try:
        resp = requests.get(f"{route}", params=[("api_key", key)] + params, timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(str(e).replace(key, "***")) from None
    time.sleep(PAUSE)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} from {route}: {resp.text[:300].replace(key, '***')}")
    return resp.json()["response"]


def demand_params(ba, kind, start=None, end=None, direction="asc", offset=0, length=PAGE):
    p = [
        ("frequency", "hourly"),
        ("data[0]", "value"),
        ("facets[respondent][]", ba),
        ("facets[type][]", kind),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", direction),
        ("offset", offset),
        ("length", length),
    ]
    if start:
        p.append(("start", start))
    if end:
        p.append(("end", end))
    return p


def fetch_all(ba, kind, start, end):
    rows, offset = [], 0
    while True:
        r = get(f"{BASE}/data/", demand_params(ba, kind, start, end, offset=offset))
        rows += r["data"]
        offset += PAGE
        if offset >= int(r["total"]):
            break
    return pd.DataFrame(rows)


def main():
    load_dotenv()

    print("=== Balancing authorities with hourly region data ===")
    r = get(f"{BASE}/facet/respondent", [])
    ids = [f["id"] for f in r["facets"]]
    print(f"  {len(ids)} respondents; test BAs present: " + ", ".join(f"{b}={b in ids}" for b in BAS))

    print("\n=== Series types and units ===")
    r = get(f"{BASE}/", [])
    print("  frequency options:", [f["id"] for f in r["frequency"]])
    print("  data columns:", list(r["data"].keys()))
    t = get(f"{BASE}/facet/type", [])
    print("  types:", {f["id"]: f.get("name") for f in t["facets"]})

    print("\n=== Earliest and latest hourly demand (type D) and day-ahead forecast (type DF) ===")
    for ba in BAS:
        for kind in ("D", "DF"):
            first = get(f"{BASE}/data/", demand_params(ba, kind, length=1))
            last = get(f"{BASE}/data/", demand_params(ba, kind, direction="desc", length=1))
            f = first["data"][0] if first["data"] else None
            l = last["data"][0] if last["data"] else None
            print(f"  {ba:5s} {kind:2s} rows={first['total']:>6}  first={f['period'] if f else None}  last={l['period'] if l else None}  units={f['value-units'] if f else None}")

    print("\n=== Spring-forward DST check, 2024-03-09 to 2024-03-11 (3 days = 72 hours if the clock is continuous) ===")
    for ba in BAS:
        df = fetch_all(ba, "D", "2024-03-09T00", "2024-03-11T23")
        p = pd.to_datetime(df["period"], format="%Y-%m-%dT%H")
        print(f"  {ba:5s} rows={len(df)}  unique periods={p.nunique()}  first={df['period'].iloc[0]}  last={df['period'].iloc[-1]}")

    print("\n=== Pagination and continuity, 2023 full year (8760 hours) ===")
    for ba in BAS:
        df = fetch_all(ba, "D", "2023-01-01T00", "2023-12-31T23")
        p = pd.to_datetime(df["period"], format="%Y-%m-%dT%H")
        full = pd.date_range(p.min(), p.max(), freq="h")
        missing = full.difference(p)
        dup = int(p.duplicated().sum())
        vals = pd.to_numeric(df["value"], errors="coerce")
        print(f"  {ba:5s} rows={len(df)}  duplicates={dup}  missing hours={len(missing)}  null values={int(vals.isna().sum())}  min={vals.min():.0f}  max={vals.max():.0f}")


if __name__ == "__main__":
    main()
