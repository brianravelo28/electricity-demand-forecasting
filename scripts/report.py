"""Print the headline tables from saved predictions, optionally leaving some BAs out.

    python scripts/report.py --variant A
    python scripts/report.py --variant A --exclude SRP

Reads data/eval/predictions_{variant}.parquet (written by run_evaluation.py), so it needs no
refit. Excluding a BA only removes its rows from scoring; the models were fit with it.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from electricity import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["A", "B"], required=True)
ap.add_argument("--exclude", default="", help="comma-separated BA codes to leave out of scoring")
args = ap.parse_args()

preds = pd.read_parquet(ROOT / "data" / "eval" / f"predictions_{args.variant}.parquet")
if args.exclude:
    preds = preds[~preds["ba"].isin(args.exclude.split(","))]
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

models = [m for m in evaluate.MODEL_ORDER if m != "eia_df"]
common = evaluate.common_sample(preds)
print(f"Variant {args.variant}, excluding [{args.exclude}]: {common['ba'].nunique()} BAs, {len(common):,} rows")
print(evaluate.score(common, models))
with_df = evaluate.common_sample(preds, with_df=True)
print(f"\nWhere EIA's forecast exists: {with_df['ba'].nunique()} BAs, {len(with_df):,} rows")
print(evaluate.score(with_df))
print("\nMAPE by year, rows where EIA's forecast exists:")
print(evaluate.mape_by(with_df, with_df["local_date"].dt.year))
