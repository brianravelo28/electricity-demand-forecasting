"""Classify the API's respondents using EIA's EIA-930 reference tables.

Downloads the reference workbook to data/raw/, joins it to the API's respondent
list and the first/last hourly demand period per respondent, and writes
data/respondents.parquet.
"""
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from eia import BASE, demand_params, get

REFERENCE_URL = "https://www.eia.gov/electricity/930-content/EIA930_Reference_Tables.xlsx"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "EIA930_Reference_Tables.xlsx"
OUT = ROOT / "data" / "respondents.parquet"


def download_reference():
    if RAW.exists():
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(REFERENCE_URL, timeout=60)
    resp.raise_for_status()
    RAW.write_bytes(resp.content)


def demand_span(code):
    first = get(f"{BASE}/data/", demand_params(code, "D", length=1))
    if not first["data"]:
        return pd.Series({"d_rows": 0, "d_first": None, "d_last": None})
    last = get(f"{BASE}/data/", demand_params(code, "D", direction="desc", length=1))
    return pd.Series({
        "d_rows": int(first["total"]),
        "d_first": first["data"][0]["period"],
        "d_last": last["data"][0]["period"],
    })


def main():
    load_dotenv()
    download_reference()
    bas = pd.read_excel(RAW, sheet_name="BAs", dtype=str).fillna("")
    regions = pd.read_excel(RAW, sheet_name="Regions", dtype=str).fillna("")

    api = pd.DataFrame(get(f"{BASE}/facet/respondent", [])["facets"])[["id", "name"]]
    api.columns = ["code", "api_name"]

    ba_cols = {"BA Code": "code", "Time Zone": "time_zone", "Region/Country Code": "region",
               "Generation Only BA": "generation_only", "U.S. BA": "us_ba", "Active BA": "active",
               "Activation Date": "activated", "Retirement Date": "retired"}
    df = api.merge(bas[list(ba_cols)].rename(columns=ba_cols), on="code", how="left")
    reg = regions.rename(columns={"Region/Country Code": "code", "Time Zone": "region_time_zone"})
    df = df.merge(reg[["code", "region_time_zone"]], on="code", how="left")

    df["kind"] = "unmatched"
    df.loc[df["time_zone"].notna(), "kind"] = "ba"
    df.loc[df["region_time_zone"].notna(), "kind"] = "region"
    df["time_zone"] = df["time_zone"].fillna(df["region_time_zone"])
    df = df.drop(columns="region_time_zone")

    spans = df["code"].apply(demand_span)
    df = pd.concat([df, spans], axis=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"respondents from API: {len(df)}")
    print(df["kind"].value_counts().to_string())
    ba = df[df["kind"] == "ba"]
    print("\nBAs by generation_only / us_ba / active:")
    print(ba.groupby(["generation_only", "us_ba", "active"]).size().to_string())
    print("\nBAs with a demand series:", int((ba["d_rows"] > 0).sum()), "of", len(ba))
    print("\nUnmatched respondents:")
    print(df.loc[df["kind"] == "unmatched", ["code", "api_name", "d_rows"]].to_string(index=False))
    print("\nTime zones among BAs with demand:")
    print(ba.loc[ba["d_rows"] > 0, "time_zone"].value_counts().to_string())


if __name__ == "__main__":
    main()
