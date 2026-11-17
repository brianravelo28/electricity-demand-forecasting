"""Bootstrap intervals for the main comparisons between forecasts.

    python scripts/bootstrap_intervals.py

For each comparison the statistic is the paired difference in mean absolute percentage error, in
percentage points (model A minus model B, so a negative value means model A is more accurate), over
the hours both forecasts cover. Whole calendar weeks are resampled with replacement, all BAs together,
because a week of weather hits many BAs at once and hours within a week are strongly related. The
interval is the 2.5th to 97.5th percentile of 2,000 resamples (seed 20261117). The by-year rows
resample only the weeks of that year.

Needs data/eval/predictions_{A,B}.parquet, extra_{A,B}_ridge.parquet and extra_A_lightgbm_noweather.parquet.
Writes results/bootstrap.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from electricity import evaluate  # noqa: E402

EVAL = ROOT / "data" / "eval"
RESAMPLES = 2000
SEED = 20261117


def load(variant):
    base = pd.read_parquet(EVAL / f"predictions_{variant}.parquet")
    base["local_date"] = pd.to_datetime(base["local_date"])
    ridge = pd.read_parquet(EVAL / f"extra_{variant}_ridge.parquet")[["ba", "stamp", "pred_ridge"]]
    noweather = pd.read_parquet(EVAL / "extra_A_lightgbm_noweather.parquet")[["ba", "stamp", "pred_lightgbm_noweather"]]
    return base.merge(ridge, on=["ba", "stamp"], how="left").merge(noweather, on=["ba", "stamp"], how="left")


def paired_difference(df, a, b):
    """Per-hour difference in absolute percentage error, in percentage points."""
    return 100 * ((df[a] - df["y"]).abs() - (df[b] - df["y"]).abs()) / df["y"]


def interval(df, a, b, rng):
    diff = paired_difference(df, a, b)
    weeks = df["local_date"].dt.to_period("W").astype(str)
    by_week = diff.groupby(weeks).agg(["sum", "count"])
    total, count = by_week["sum"].to_numpy(), by_week["count"].to_numpy()
    pick = rng.integers(0, len(total), size=(RESAMPLES, len(total)))
    stats = total[pick].sum(axis=1) / count[pick].sum(axis=1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return total.sum() / count.sum(), low, high, len(diff), len(total)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for variant in "AB":
        data = load(variant)
        lgbm_col = "pred_lightgbm"
        samples = {
            "20 BAs": evaluate.common_sample(data),
            "17 BAs with an EIA forecast": evaluate.common_sample(data, with_df=True),
            "16 BAs with an EIA forecast, without SRP": evaluate.common_sample(data[data["ba"] != "SRP"], with_df=True),
        }
        comparisons = [
            ("LightGBM minus yesterday's demand", "pred_lightgbm", "pred_previous_day_naive", "20 BAs"),
            ("LightGBM minus regression", "pred_lightgbm", "pred_regression", "20 BAs"),
            ("LightGBM minus ridge", "pred_lightgbm", "pred_ridge", "20 BAs"),
            ("Ridge minus yesterday's demand", "pred_ridge", "pred_previous_day_naive", "20 BAs"),
            ("LightGBM minus LightGBM without weather", "pred_lightgbm", "pred_lightgbm_noweather", "20 BAs"),
            ("LightGBM minus EIA forecast", "pred_lightgbm", "eia_df", "17 BAs with an EIA forecast"),
            ("Ridge minus EIA forecast", "pred_ridge", "eia_df", "17 BAs with an EIA forecast"),
            ("LightGBM minus EIA forecast", "pred_lightgbm", "eia_df", "16 BAs with an EIA forecast, without SRP"),
        ]
        for name, a, b, sample in comparisons:
            df = samples[sample].dropna(subset=[a, b])
            periods = [("2021 to 2026", df)]
            if "EIA" in name or name.startswith("LightGBM minus ridge"):
                periods += [(str(y), df[df["local_date"].dt.year == y]) for y in range(2021, 2027)]
            for period, part in periods:
                diff, low, high, hours, weeks = interval(part, a, b, rng)
                rows.append(dict(variant=variant, comparison=name, sample=sample, period=period, hours=hours, weeks=weeks,
                                 difference_points=round(diff, 3), ci_low=round(low, 3), ci_high=round(high, 3),
                                 interval_excludes_zero=bool(low > 0 or high < 0)))
        print(f"variant {variant}: done", flush=True)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "bootstrap.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(table[table["period"] == "2021 to 2026"].to_string(index=False))


if __name__ == "__main__":
    main()
