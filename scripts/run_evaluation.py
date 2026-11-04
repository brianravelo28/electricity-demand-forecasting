"""Run the walk-forward evaluation and print the results tables.

    python scripts/run_evaluation.py --variant A
    python scripts/run_evaluation.py --variant B --bas ERCO,PJM --max-blocks 2   # quick trial

Writes predictions and tables to data/eval/ (git-ignored).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `electricity` imports when run as a script

from ba_timezones import BA_TIMEZONE  # noqa: E402
from electricity import evaluate  # noqa: E402

OUT = ROOT / "data" / "eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["A", "B"], required=True)
    ap.add_argument("--bas", default=",".join(BA_TIMEZONE), help="comma-separated BA codes")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--max-blocks", type=int, default=None)
    args = ap.parse_args()
    bas = args.bas.split(",")
    tag = f"{args.variant}" + ("_trial" if args.max_blocks or args.bas != ",".join(BA_TIMEZONE) else "")

    t0 = time.time()
    rows = evaluate.load_rows(args.variant, bas, BA_TIMEZONE)
    print(f"rows: {len(rows):,}  BAs: {len(bas)}  loaded in {time.time() - t0:.0f}s", flush=True)
    preds = evaluate.walk_forward(rows, bas, args.variant, args.start, args.rounds, args.max_blocks,
                                  log=lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True))
    OUT.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(OUT / f"predictions_{tag}.parquet", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    common = evaluate.common_sample(preds)
    print(f"\nTest rows: {len(preds):,}; scored on the {len(common):,} rows where every model has a forecast")
    overall = evaluate.score(common, [m for m in evaluate.MODEL_ORDER if m != "eia_df"])
    print("\nOverall (percent):\n", overall)

    with_df = evaluate.common_sample(preds, with_df=True)
    print(f"\nWhere EIA's day-ahead forecast exists ({len(with_df):,} rows, FPC, PSCO and LGEE excluded):\n", evaluate.score(with_df))
    print("\nMAPE by BA (percent):\n", evaluate.mape_by(common, "ba"))
    print("\nMAPE by year (percent):\n", evaluate.mape_by(common, common["local_date"].dt.year))
    print("\nMAPE by regime (percent):\n", evaluate.regime_table(common))

    overall.to_csv(OUT / f"overall_{tag}.csv")
    evaluate.mape_by(common, "ba").to_csv(OUT / f"by_ba_{tag}.csv")
    evaluate.regime_table(common).to_csv(OUT / f"by_regime_{tag}.csv")
    print(f"\nTotal time {time.time() - t0:.0f}s; files in {OUT}")


if __name__ == "__main__":
    main()
