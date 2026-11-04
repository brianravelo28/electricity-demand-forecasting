"""Rerun the evaluation with different cleaning thresholds to see if the model ranking moves.

    python scripts/run_sensitivity.py --low 0.3 --high 3.0
    python scripts/run_sensitivity.py --low 0.5 --high 1.8

Variant A only. The cleaning thresholds are module constants read at call time, so they are
overridden here without touching the library code. Outputs go to data/eval/ with a tag.
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
from electricity import cleaning, evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--low", type=float, required=True)
ap.add_argument("--high", type=float, required=True)
ap.add_argument("--rounds", type=int, default=400)
args = ap.parse_args()
cleaning.LOW, cleaning.HIGH = args.low, args.high
tag = f"A_clean_{args.low}_{args.high}"

t0 = time.time()
bas = list(BA_TIMEZONE)
rows = evaluate.load_rows("A", bas, BA_TIMEZONE)
preds = evaluate.walk_forward(rows, bas, "A", "2021-01-01", args.rounds,
                              log=lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True))
out = ROOT / "data" / "eval"
preds.to_parquet(out / f"predictions_{tag}.parquet", index=False)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
common = evaluate.common_sample(preds)
score = evaluate.score(common, [m for m in evaluate.MODEL_ORDER if m != "eia_df"])
print(f"\nThresholds {args.low} / {args.high}; {len(common):,} scored rows\n", score)
score.to_csv(out / f"overall_{tag}.csv")
print("\nMAPE by regime:\n", evaluate.regime_table(common))
