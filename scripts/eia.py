"""Shared helpers for the EIA API v2 hourly grid data.

The key travels in the query string, so errors are scrubbed before they are shown.
"""
import os
import time

import pandas as pd
import requests

BASE = "https://api.eia.gov/v2/electricity/rto/region-data"
PAGE = 5000
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
