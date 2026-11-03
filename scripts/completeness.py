"""Completeness of hourly demand for 2019 to 2020 per BA, and the draft selection rule.

Reads data/raw/hourly/*_D.parquet and data/respondents.parquet; writes
data/ba_completeness.csv.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "hourly"
START = pd.Timestamp("2019-01-01T00", tz="UTC")
END = pd.Timestamp("2020-12-31T23", tz="UTC")
MAX_BAD_SHARE = 0.02


def main():
    meta = pd.read_parquet(ROOT / "data" / "respondents.parquet").set_index("code")
    expected = len(pd.date_range(START, END, freq="h"))

    rows = []
    for f in sorted(RAW.glob("*_D.parquet")):
        df = pd.read_parquet(f)
        code = df["respondent"].iloc[0]
        w = df[(df["period"] >= START) & (df["period"] <= END)]
        unique = w["period"].nunique()
        valid = w["value"].where(w["value"] > 0)
        missing = expected - unique
        nulls = int(w["value"].isna().sum())
        nonpos = int((w["value"] <= 0).sum())
        rows.append({
            "code": code,
            "time_zone": meta.loc[code, "time_zone"],
            "first": df["period"].min(),
            "last": df["period"].max(),
            "missing_hours": missing,
            "null_values": nulls,
            "nonpositive": nonpos,
            "duplicates": len(w) - unique,
            "bad_share": (missing + nulls + nonpos) / expected,
            "mean_mw": valid.mean(),
        })
    t = pd.DataFrame(rows)

    t["full_history"] = t["first"] <= START
    t["eligible"] = t["full_history"] & (t["bad_share"] <= MAX_BAD_SHARE)
    t = t.sort_values("mean_mw", ascending=False).reset_index(drop=True)
    t.to_csv(ROOT / "data" / "ba_completeness.csv", index=False)

    pd.set_option("display.width", 200)
    show = ["code", "time_zone", "mean_mw", "missing_hours", "null_values", "nonpositive", "bad_share"]
    print(f"BAs downloaded: {len(t)}   expected hours 2019-2020: {expected}")
    print(f"eligible (data from {START.date()}, bad share <= {MAX_BAD_SHARE:.0%}): {int(t['eligible'].sum())}")
    print(f"duplicate periods in window: {int(t['duplicates'].sum())}\n")
    print("Eligible, ranked by mean demand 2019-2020:")
    print(t.loc[t["eligible"], show].round({"mean_mw": 0, "bad_share": 4}).to_string())
    print("\nNot eligible:")
    bad = t.loc[~t["eligible"], show + ["full_history"]].round({"mean_mw": 0, "bad_share": 4})
    print(bad.to_string())


if __name__ == "__main__":
    main()
