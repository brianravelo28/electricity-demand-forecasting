"""Draft mapping from BAs to NOAA ISD weather stations (airports in each load center).

Resolves ICAO codes to ISD station ids using data/raw/weather/isd-history.csv and
checks each station reports from 2019 through the last weeks. Writes
data/weather_stations.parquet. The airport choices are a judgement call and equal
weighted; they are a draft to review.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "data" / "raw" / "weather" / "isd-history.csv"
OUT = ROOT / "data" / "weather_stations.parquet"

BA_AIRPORTS = {
    "PJM": ["KPHL", "KDCA", "KORD", "KPIT"],
    "MISO": ["KMSP", "KSTL", "KDTW", "KMSY"],
    "ERCO": ["KDFW", "KIAH", "KAUS", "KSAT"],
    "SWPP": ["KOKC", "KMCI", "KOMA"],
    "SOCO": ["KATL", "KBHM"],
    "CISO": ["KLAX", "KSFO", "KSAN"],
    "TVA": ["KBNA", "KMEM", "KTYS"],
    "NYIS": ["KJFK", "KBUF", "KALB"],
    "FPL": ["KMIA", "KFLL"],  # Palm Beach (KPBI) is not in the Iowa Mesonet archive
    "ISNE": ["KBOS", "KBDL"],
    "DUK": ["KCLT", "KGSO"],
    "CPLE": ["KRDU"],
    "BPAT": ["KPDX", "KSEA"],
    "FPC": ["KMCO", "KTPA"],
    "PSCO": ["KDEN"],
    "PACE": ["KSLC"],
    "NEVP": ["KLAS"],
    "LGEE": ["KSDF"],
    "SRP": ["KPHX"],
    "AZPS": ["KPHX"],
}

FIRST_NEEDED = "20190101"
# isd-history.csv is updated rarely (its latest end date is late August 2025), so this only
# screens out stations that stopped reporting; real coverage is checked on the data files.
LAST_NEEDED = "20250801"


def main():
    h = pd.read_csv(HISTORY, dtype=str).fillna("")
    h["station"] = h["USAF"] + h["WBAN"]
    wanted = sorted({a for v in BA_AIRPORTS.values() for a in v})

    rows = []
    for icao in wanted:
        c = h[h["ICAO"] == icao]
        ok = c[(c["BEGIN"] <= FIRST_NEEDED) & (c["END"] >= LAST_NEEDED)]
        pick = ok.sort_values("END").iloc[-1] if len(ok) else None
        rows.append({
            "icao": icao,
            "candidates": len(c),
            "station": pick["station"] if pick is not None else None,
            "name": pick["STATION NAME"] if pick is not None else None,
            "state": pick["STATE"] if pick is not None else None,
            "begin": pick["BEGIN"] if pick is not None else None,
            "end": pick["END"] if pick is not None else None,
        })
    st = pd.DataFrame(rows)

    links = pd.DataFrame([(ba, a) for ba, v in BA_AIRPORTS.items() for a in v], columns=["ba", "icao"])
    out = links.merge(st, on="icao", how="left")
    out.to_parquet(OUT, index=False)

    pd.set_option("display.width", 200)
    print(f"airports: {len(st)}   resolved: {int(st['station'].notna().sum())}")
    print(st.to_string(index=False))
    miss = out[out["station"].isna()]
    print("\nBAs with an unresolved airport:", sorted(miss["ba"].unique()) if len(miss) else "none")


if __name__ == "__main__":
    main()
