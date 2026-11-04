"""Check the cleaning rule's removals against EIA's day-ahead forecast (DF).

A genuine reading should sit near the forecast for that hour; a corrupt one should not.
For every removed reading with a forecast, print how far it is from the forecast, and
compare with the typical gap on kept readings. Removals that match the forecast closely
are the ones to worry about (real data being thrown away).

    python scripts/check_cleaning.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ba_timezones import BA_TIMEZONE  # noqa: E402
from electricity import cleaning  # noqa: E402
from electricity.cleaning import clean_demand  # noqa: E402

if len(sys.argv) == 3:  # optional: python scripts/check_cleaning.py LOW HIGH
    cleaning.LOW, cleaning.HIGH = float(sys.argv[1]), float(sys.argv[2])
print(f"Thresholds {cleaning.LOW} / {cleaning.HIGH}")
pd.set_option("display.width", 200)
rows, close = [], []
for ba in BA_TIMEZONE:
    d = pd.read_parquet(ROOT / "data" / "raw" / "hourly" / f"{ba}_D.parquet").set_index("period")["value"].sort_index()
    df = pd.read_parquet(ROOT / "data" / "raw" / "hourly" / f"{ba}_DF.parquet").set_index("period")["value"].sort_index()
    d = d[~d.index.duplicated()]
    df = df[~df.index.duplicated()]
    _, bad = clean_demand(d)
    gap = ((d - df.reindex(d.index)) / df.reindex(d.index)).abs()
    kept, removed = gap[~bad].dropna(), gap[bad].dropna()
    rows.append({"ba": ba, "removed": int(bad.sum()), "with_forecast": len(removed),
                 "median_gap_kept_%": 100 * kept.median(), "median_gap_removed_%": 100 * removed.median() if len(removed) else None,
                 "removed_within_25%_of_forecast": int((removed < 0.25).sum())})
    for t, g in removed[removed < 0.25].items():
        close.append({"ba": ba, "stamp": t, "demand": d[t], "forecast": df[t], "gap_%": 100 * g})
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\nRemoved readings that sit within 25% of EIA's forecast (possibly genuine):")
print(pd.DataFrame(close).round({"gap_%": 1}).to_string(index=False) if close else "none")
