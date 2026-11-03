"""Download raw hourly series for every BA with a demand series to parquet.

One file per BA and type in data/raw/hourly/. Reads data/respondents.parquet (from
build_respondents.py). Existing files are skipped, so an interrupted run resumes.
Everything downstream reads only these files.

    python scripts/ingest.py --types D
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from eia import fetch_all

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "hourly"
START = "2019-01-01T00"


def fetch_with_retry(code, kind, tries=4):
    for i in range(tries):
        try:
            return fetch_all(code, kind, START, None)
        except RuntimeError as e:
            if i == tries - 1 or not any(s in str(e) for s in ("HTTP 429", "HTTP 5")):
                raise
            wait = 30 * 2 ** i
            print(f"  retrying in {wait}s: {str(e)[:80]}", flush=True)
            time.sleep(wait)


def save(df, path):
    out = pd.DataFrame({
        "respondent": df["respondent"],
        "type": df["type"],
        "period": pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True),
        "value": pd.to_numeric(df["value"], errors="coerce"),
    })
    tmp = path.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    return len(out)


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="D", help="comma-separated EIA types, e.g. D,DF")
    args = ap.parse_args()

    meta = pd.read_parquet(ROOT / "data" / "respondents.parquet")
    codes = sorted(meta.loc[(meta["kind"] == "ba") & (meta["d_rows"] > 0), "code"])
    RAW.mkdir(parents=True, exist_ok=True)

    for kind in args.types.split(","):
        for n, code in enumerate(codes, 1):
            path = RAW / f"{code}_{kind}.parquet"
            if path.exists():
                continue
            df = fetch_with_retry(code, kind)
            rows = save(df, path) if len(df) else 0
            print(f"[{kind}] {n}/{len(codes)} {code}: {rows} rows", flush=True)


if __name__ == "__main__":
    main()
