"""Check each BA's civil time zone against its demand profile.

Two tests on weekday mean demand (UTC hours):
  1. the winter morning ramp (steepest rise from 03:00 to 11:00 local standard time) should
     land near 07:00 local under the chosen zone;
  2. the ramp should move one UTC hour earlier after the March clock change for BAs that
     observe daylight saving, and not move for Phoenix.
Reads data/raw/hourly/*_D.parquet.
"""
import warnings
from pathlib import Path

import pandas as pd

from ba_timezones import BA_TIMEZONE

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "hourly"
STANDARD_OFFSET = {"America/New_York": -5, "America/Chicago": -6, "America/Denver": -7,
                   "America/Phoenix": -7, "America/Los_Angeles": -8}
SWITCH_DATES = ["2022-03-13", "2023-03-12", "2024-03-10", "2025-03-09"]


def normalised_profile(df, start, end):
    s = df[(df["period"] >= start) & (df["period"] < end) & (df["period"].dt.day_of_week < 5)]
    p = s.groupby(s["period"].dt.hour)["value"].mean()
    return (p - p.min()) / (p.max() - p.min())


def ramp_hour(profile, lo, hi):
    return int(profile.diff().reindex(range(lo, hi + 1)).idxmax())


def main():
    warnings.filterwarnings("ignore")
    day = pd.Timedelta(days=1)
    rows = []
    for ba, tz in BA_TIMEZONE.items():
        off = STANDARD_OFFSET[tz]
        lo, hi = 3 - off, 11 - off
        df = pd.read_parquet(RAW / f"{ba}_D.parquet").dropna(subset=["value"])

        winter = pd.concat([normalised_profile(df, pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(f"{y}-03-01", tz="UTC"))
                            for y in (2022, 2023, 2024, 2025)], axis=1).mean(axis=1)
        before, after = [], []
        for d in SWITCH_DATES:
            d0 = pd.Timestamp(d, tz="UTC")
            before.append(normalised_profile(df, d0 - 14 * day, d0 - day))
            after.append(normalised_profile(df, d0 + 2 * day, d0 + 15 * day))
        rb = ramp_hour(pd.concat(before, axis=1).mean(axis=1), lo, hi)
        ra = ramp_hour(pd.concat(after, axis=1).mean(axis=1), lo, hi)
        expected = 0 if tz == "America/Phoenix" else -1
        rows.append({"ba": ba, "zone": tz, "winter_ramp_local": (ramp_hour(winter, lo, hi) + off) % 24,
                     "ramp_before_utc": rb, "ramp_after_utc": ra, "shift": ra - rb, "expected_shift": expected,
                     "ok": (ra - rb) == expected})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))
    print(f"\nDaylight-saving shift as expected for {int(t['ok'].sum())} of {len(t)} BAs")


if __name__ == "__main__":
    main()
