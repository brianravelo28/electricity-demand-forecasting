"""SHAP importance for the LightGBM model, computed with LightGBM's built-in tree SHAP.

    python scripts/run_shap.py --variant A

Fits one model on origin dates up to two days before the test year (same rule as the
walk-forward) and explains its forecasts for that whole year, so the rows are out of
sample and cover every season. The target and demand inputs are scaled by each BA's
training-window mean, so a SHAP value of 0.05 means the feature moved the forecast by
5% of that BA's average demand. Writes data/eval/shap_{variant}.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ba_timezones import BA_TIMEZONE  # noqa: E402
from electricity import evaluate, models  # noqa: E402
from electricity.features import split_by_origin  # noqa: E402

GROUPS = {
    "Recent demand and lags": models.LEVEL_COLS,
    "Calendar and holidays": models.CALENDAR_COLS,
    "Weather before the forecast": models.WEATHER_A,
    "Target-day weather (variant B only)": models.WEATHER_B,
    "Balancing authority": ["ba"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["A", "B"], required=True)
ap.add_argument("--year", type=int, default=2025)
ap.add_argument("--sample", type=int, default=60000)
ap.add_argument("--rounds", type=int, default=400)
args = ap.parse_args()

bas = list(BA_TIMEZONE)
rows = evaluate.load_rows(args.variant, bas, BA_TIMEZONE)
train, test = split_by_origin(rows, f"{args.year}-01-01", f"{args.year}-12-31")
test = test.dropna(subset=["y"])
print(f"train rows {len(train):,}; test rows {len(test):,}", flush=True)

_, booster = models.fit_predict_lgbm(train, test.head(1), args.variant, bas, args.rounds)
sample = test.sample(min(args.sample, len(test)), random_state=0)
scale = models.training_scale(train)
x, _ = models._matrix(sample, scale, bas, args.variant)
contrib = booster.predict(x, pred_contrib=True)
shap = pd.DataFrame(contrib[:, :-1], columns=x.columns)
mean_abs = shap.abs().mean().mul(100).rename("mean_abs_shap_pct_of_BA_demand").sort_values(ascending=False)

out = ROOT / "data" / "eval"
mean_abs.to_csv(out / f"shap_{args.variant}.csv")
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
print(f"\nVariant {args.variant}, model fit through {args.year - 1}, explained on {len(sample):,} rows of {args.year}")
print("\nTop 15 features (mean |SHAP|, percent of the BA's average demand):\n", mean_abs.head(15).to_string())
print("\nBy group (sum of mean |SHAP|; correlated features share credit, so read groups first):")
for name, cols in GROUPS.items():
    cols = [c for c in cols if c in mean_abs.index]
    if cols:
        print(f"  {name}: {mean_abs[cols].sum():.3f}  ({len(cols)} features)")
print("  total:", round(float(mean_abs.sum()), 3))
