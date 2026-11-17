"""Walk-forward forecasts from two extra comparison models, saved for the comparison tables.

    python scripts/run_extra_models.py --variant A --models ridge
    python scripts/run_extra_models.py --variant B --models ridge
    python scripts/run_extra_models.py --variant A --models lightgbm_noweather

ridge is a ridge regression on the same inputs as LightGBM; lightgbm_noweather is LightGBM with no
weather inputs (it does not depend on the weather variant, so run it once). Same quarterly blocks and
training rule as run_evaluation.py. Writes data/eval/extra_{variant}_{model}.parquet.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ba_timezones import BA_TIMEZONE  # noqa: E402
from electricity import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["A", "B"], required=True)
ap.add_argument("--models", required=True, help="comma-separated: ridge, lightgbm_noweather")
ap.add_argument("--start", default="2021-01-01")
ap.add_argument("--rounds", type=int, default=400)
args = ap.parse_args()

t0 = time.time()
bas = list(BA_TIMEZONE)
rows = evaluate.load_rows(args.variant, bas, BA_TIMEZONE)
for name in args.models.split(","):
    out = evaluate.walk_forward_extra(rows, bas, args.variant, args.start, args.rounds, which=(name,),
                                      log=lambda s: print(f"[{time.time() - t0:5.0f}s] {name}: {s}", flush=True))
    path = ROOT / "data" / "eval" / f"extra_{args.variant}_{name}.parquet"
    out.to_parquet(path, index=False)
    print(f"{name}: {len(out):,} rows written to {path} after {time.time() - t0:.0f}s", flush=True)
