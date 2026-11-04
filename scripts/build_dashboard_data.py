"""Build the small JSON files the dashboard reads, from the saved evaluation outputs.

    python scripts/build_dashboard_data.py

Reads data/eval/predictions_A.parquet and predictions_B.parquet (from run_evaluation.py),
data/eval/shap_A.csv and shap_B.csv (from run_shap.py) and data/respondents.parquet.
Writes data/dashboard/*.json. Nothing here calls an API. Everything in these files is
computed by the author from EIA data, except the `eia_forecast` columns, which are the
forecasts BAs report to EIA, unaltered.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from electricity import evaluate  # noqa: E402
from electricity import models as models_mod  # noqa: E402

OUT = ROOT / "data" / "dashboard"
EVAL = ROOT / "data" / "eval"
MODELS = ["seasonal_naive", "previous_day_naive", "same_day_avg", "regression", "lightgbm", "eia_df"]
NO_EIA = [m for m in MODELS if m != "eia_df"]
SAMPLES = {
    "all": ("All 20 BAs", None),
    "eia": ("17 BAs that have an EIA forecast", None),
    "all_no_srp": ("19 BAs, without SRP", "SRP"),
    "eia_no_srp": ("16 BAs with an EIA forecast, without SRP", "SRP"),
}
WINDOWS = {
    "uri": ("Winter Storm Uri", "2021-02-13", "2021-02-19"),
    "texas_heat": ("Texas heat wave, August 2023", "2023-08-07", "2023-08-13"),
    "elliott": ("Winter Storm Elliott", "2022-12-21", "2022-12-27"),
    "spring": ("An ordinary spring week, 2025", "2025-04-14", "2025-04-20"),
    "latest": ("Latest full week of data", "2026-10-19", "2026-10-25"),
}
FEATURE_LABELS = {
    "lag_day": "Demand at this hour yesterday",
    "lag_week_1": "Demand at this hour last week",
    "lag_week_2": "Demand at this hour two weeks ago",
    "lag_week_3": "Demand at this hour three weeks ago",
    "lag_week_4": "Demand at this hour four weeks ago",
    "demand_last": "Most recent demand reading",
    "demand_mean_168h": "Average demand, past week",
    "demand_mean_24h": "Average demand, past day",
    "demand_max_24h": "Peak demand, past day",
    "demand_change_3h": "Demand change, past 3 hours",
    "temp_last": "Most recent temperature",
    "temp_mean_24h": "Average temperature, past day",
    "temp_change_3h": "Temperature change, past 3 hours",
    "dew_last": "Most recent dew point",
    "wind_last": "Most recent wind speed",
    "temp_target": "Temperature at the target hour",
    "dew_target": "Dew point at the target hour",
    "wind_target": "Wind speed at the target hour",
    "temp_day_max": "Highest temperature on the target day",
    "temp_day_min": "Lowest temperature on the target day",
    "temp_day_mean": "Average temperature on the target day",
    "hdd_day": "Heating degrees, target day",
    "cdd_day": "Cooling degrees, target day",
    "weekday": "Day of the week",
    "hour_local": "Hour of the day",
    "doy_sin": "Time of year (sine)",
    "doy_cos": "Time of year (cosine)",
    "month": "Month",
    "horizon": "Hours ahead",
    "is_holiday": "Holiday",
    "day_before_holiday": "Day before a holiday",
    "day_after_holiday": "Day after a holiday",
    "hours_in_day": "Hours in the day (daylight saving)",
    "lag_day_age": "Age of the yesterday reading",
    "ba": "Which balancing authority",
}


def select(preds, sample, with_df):
    out = preds
    drop = SAMPLES[sample][1]
    if drop:
        out = out[out["ba"] != drop]
    return evaluate.common_sample(out, with_df=with_df)


def mape(df, model):
    return float(100 * ((df[f"pred_{model}"] - df["y"]) / df["y"]).abs().mean())


def num(x):
    """Whole MW, or None when the value is missing."""
    return None if pd.isna(x) else round(float(x))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    resp = pd.read_parquet(ROOT / "data" / "respondents.parquet")
    names = dict(zip(resp["code"], resp["api_name"]))  # EIA's name for each balancing authority
    preds = {v: pd.read_parquet(EVAL / f"predictions_{v}.parquet") for v in "AB"}

    overall, by_year, regimes = [], [], []
    for sample in SAMPLES:
        with_df = sample.startswith("eia")
        for v, p in preds.items():
            c = select(p, sample, with_df)
            models = MODELS if with_df else NO_EIA
            nba = int(c["ba"].nunique())
            for m in models:
                rel = (c[f"pred_{m}"] - c["y"]) / c["y"]
                overall.append(dict(sample=sample, variant=v, model=m, n_bas=nba, hours=int(len(c)),
                                    mape=round(100 * float(rel.abs().mean()), 3),
                                    peak_mape=round(100 * float(rel[c["is_peak"]].abs().mean()), 3),
                                    bias=round(100 * float(rel.mean()), 3)))
            yr = c["local_date"].dt.year
            for m in models:
                err = (100 * ((c[f"pred_{m}"] - c["y"]) / c["y"]).abs()).groupby(yr).mean()
                for y, val in err.items():
                    by_year.append(dict(sample=sample, variant=v, year=int(y), model=m, mape=round(float(val), 3)))
            tagged = np.zeros(len(c), dtype=bool)
            for name, rule in evaluate.REGIMES.items():
                mask = rule(c).to_numpy()
                tagged |= mask
                for m in models:
                    regimes.append(dict(sample=sample, variant=v, regime=name, model=m, hours=int(mask.sum()),
                                        mape=round(mape(c[mask], m), 3)))
            for m in models:
                regimes.append(dict(sample=sample, variant=v, regime="Calm (none of the above)", model=m,
                                    hours=int((~tagged).sum()), mape=round(mape(c[~tagged], m), 3)))

    by_ba = []
    for v, p in preds.items():
        c = evaluate.common_sample(p)
        for ba, g in c.groupby("ba"):
            for m in NO_EIA:
                by_ba.append(dict(ba=ba, name=names.get(ba, ba), variant=v, model=m, mape=round(mape(g, m), 3)))
        w = evaluate.common_sample(p, with_df=True)
        for ba, g in w.groupby("ba"):
            by_ba.append(dict(ba=ba, name=names.get(ba, ba), variant=v, model="eia_df", mape=round(mape(g, "eia_df"), 3)))

    monthly = []
    for v, p in preds.items():
        c = evaluate.common_sample(p)
        c = c.assign(month=c["local_date"].dt.strftime("%Y-%m"))
        for m in NO_EIA:
            e = (100 * ((c[f"pred_{m}"] - c["y"]) / c["y"]).abs()).groupby([c["ba"], c["month"]]).mean()
            monthly += [dict(ba=b, month=mo, variant=v, model=m, mape=round(float(x), 3)) for (b, mo), x in e.items()]
        w = evaluate.common_sample(p, with_df=True)
        w = w.assign(month=w["local_date"].dt.strftime("%Y-%m"))
        e = (100 * ((w["pred_eia_df"] - w["y"]) / w["y"]).abs()).groupby([w["ba"], w["month"]]).mean()
        monthly += [dict(ba=b, month=mo, variant=v, model="eia_df", mape=round(float(x), 3)) for (b, mo), x in e.items()]

    # Hourly windows. One row per BA and hour; forecasts from both weather variants side by side.
    a, b = preds["A"], preds["B"]
    b = b[["ba", "stamp", "pred_lightgbm"]].rename(columns={"pred_lightgbm": "lightgbm_b"})
    hourly = []
    for wid, (label, start, end) in WINDOWS.items():
        w = a[(a["local_date"] >= start) & (a["local_date"] <= end)]
        w = w.merge(b, on=["ba", "stamp"], how="left").sort_values(["ba", "stamp"])
        for ba, g in w.groupby("ba"):
            for i, r in enumerate(g.itertuples()):
                hourly.append(dict(
                    ba=ba, window=wid, i=i,
                    local=f"{r.local_date:%Y-%m-%d} {int(r.hour_local):02d}:00",
                    actual=num(r.y),
                    lightgbm_a=num(r.pred_lightgbm),
                    lightgbm_b=num(r.lightgbm_b),
                    previous_day=num(r.pred_previous_day_naive),
                    eia_forecast=num(r.eia_df),
                ))

    GROUPS = {  # same grouping as scripts/run_shap.py
        "Recent demand and lags": models_mod.LEVEL_COLS,
        "Calendar and holidays": models_mod.CALENDAR_COLS,
        "Weather known at the forecast time": models_mod.WEATHER_A,
        "Target-day weather (variant B only)": models_mod.WEATHER_B,
        "Which balancing authority": ["ba"],
    }
    shap = []
    for v in "AB":
        s = pd.read_csv(EVAL / f"shap_{v}.csv", index_col=0).iloc[:, 0]
        total = float(s.sum())
        for g, cols in GROUPS.items():
            cols = [c for c in cols if c in s.index]
            if cols:
                shap.append(dict(variant=v, kind="group", name=g, value=round(float(s[cols].sum()), 3),
                                 share=round(100 * float(s[cols].sum()) / total, 1)))
        for f, val in s.head(12).items():
            shap.append(dict(variant=v, kind="feature", name=FEATURE_LABELS.get(f, f), value=round(float(val), 3),
                             share=round(100 * float(val) / total, 1)))

    windows = [dict(window=k, label=v[0], start=v[1], end=v[2]) for k, v in WINDOWS.items()]
    bas = [dict(ba=ba, name=names.get(ba, ba)) for ba in sorted(a["ba"].unique())]
    samples = [dict(sample=k, label=v[0]) for k, v in SAMPLES.items()]
    models = [dict(model=m, label=l, short=s) for m, l, s in [
        ("seasonal_naive", "Same hour last week", "Last week"),
        ("previous_day_naive", "Same hour yesterday", "Yesterday"),
        ("same_day_avg", "Average of the last four same weekdays", "4-week average"),
        ("regression", "Regression on calendar and temperature", "Regression"),
        ("lightgbm", "LightGBM", "LightGBM"),
        ("eia_df", "EIA day-ahead forecast", "EIA forecast")]]

    files = dict(overall=overall, by_year=by_year, regimes=regimes, by_ba=by_ba, monthly=monthly, hourly=hourly,
                 shap=shap, windows=windows, bas=bas, samples=samples, models=models)
    for name, rows in files.items():
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
        print(f"{name}.json: {len(rows):,} rows, {path.stat().st_size / 1024:,.0f} KB")


if __name__ == "__main__":
    main()
