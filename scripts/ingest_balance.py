"""Download net generation (NG) and total interchange (TI) for chosen BAs.

Same files and format as ingest.py (data/raw/hourly/{BA}_{type}.parquet), for a few BAs
only. Existing files are skipped.

    python scripts/ingest_balance.py --bas SRP,AZPS --types NG,TI
"""
import argparse

from dotenv import load_dotenv

from ingest import RAW, fetch_with_retry, save


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--bas", required=True, help="comma-separated BA codes")
    ap.add_argument("--types", default="NG,TI")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for code in args.bas.split(","):
        for kind in args.types.split(","):
            path = RAW / f"{code}_{kind}.parquet"
            if path.exists():
                continue
            df = fetch_with_retry(code, kind)
            rows = save(df, path) if len(df) else 0
            print(f"[{kind}] {code}: {rows} rows", flush=True)


if __name__ == "__main__":
    main()
