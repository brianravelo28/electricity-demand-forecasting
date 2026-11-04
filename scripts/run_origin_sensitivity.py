"""Rerun the evaluation with a different forecast origin hour to see how much it matters.

    python scripts/run_origin_sensitivity.py --origin 6 --start 2024-01-01

The demand cutoff is two hours before the origin and the weather cutoff one hour before, as
in the main run. The constants in electricity.features are read at call time, so they are
overridden here without touching the library. Variant A only. Because test blocks are
calendar quarters, a later --start gives the same blocks and models as the main run, so the
10:00 results in data/eval/predictions_A.parquet can be compared on identical rows.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ba_timezones import BA_TIMEZONE  # noqa: E402
from electricity import evaluate, features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--origin", type=int, required=True, help="origin hour, local time")
ap.add_argument("--start", default="2024-01-01")
ap.add_argument("--rounds", type=int, default=400)
args = ap.parse_args()
features.ORIGIN_HOUR = args.origin
features.LAST_DEMAND_HOUR = args.origin - 2
features.LAST_WEATHER_HOUR = args.origin - 1
tag = f"A_origin{args.origin}"

t0 = time.time()
bas = list(BA_TIMEZONE)
rows = evaluate.load_rows("A", bas, BA_TIMEZONE)
preds = evaluate.walk_forward(rows, bas, "A", args.start, args.rounds,
                              log=lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True))
out = ROOT / "data" / "eval"
preds.to_parquet(out / f"predictions_{tag}.parquet", index=False)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
common = evaluate.common_sample(preds)
print(f"\nOrigin {args.origin}:00, {len(common):,} scored rows\n",
      evaluate.score(common, [m for m in evaluate.MODEL_ORDER if m != "eia_df"]))
