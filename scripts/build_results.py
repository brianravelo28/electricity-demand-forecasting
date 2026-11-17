"""Write the committed results folder (results/) from the saved evaluation outputs and raw data.

    python scripts/build_results.py

Everything the docs quote is written here as small CSV files, so a reader can check a figure without
rerunning the pipeline. results/daily_errors.parquet holds per-BA, per-day error sums, enough to
recompute every headline table (see results/README.md). Needs data/eval/ (run_evaluation.py,
run_extra_models.py, run_shap.py, the sensitivity runs) and data/raw/. Run bootstrap_intervals.py
as well for results/bootstrap.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from electricity import evaluate, models  # noqa: E402
from electricity.cleaning import clean_demand  # noqa: E402

EVAL = ROOT / "data" / "eval"
RAW = ROOT / "data" / "raw" / "hourly"
OUT = ROOT / "results"
M5 = ["seasonal_naive", "previous_day_naive", "same_day_avg", "regression", "lightgbm"]
BAS = "PJM MISO ERCO SWPP SOCO CISO TVA NYIS FPL ISNE DUK CPLE BPAT FPC PSCO PACE NEVP LGEE SRP AZPS".split()
EIA_BAS = [b for b in BAS if b not in evaluate.DF_EXCLUDED]
SAMPLES = {  # name: (BAs left out, needs an EIA forecast)
    "all_20": ((), False), "eia_17": ((), True), "all_19_without_SRP": (("SRP",), False), "eia_16_without_SRP": (("SRP",), True),
}


def preds(tag):
    p = pd.read_parquet(EVAL / f"predictions_{tag}.parquet")
    p["local_date"] = pd.to_datetime(p["local_date"])
    return p


def sample(p, drop=(), eia=False):
    return evaluate.common_sample(p[~p["ba"].isin(drop)], with_df=eia)


def ape(df, col):
    return 100 * (df[col] - df["y"]).abs() / df["y"]


def mape(df, col):
    return float(ape(df, col).mean())


def raw(ba, kind):
    d = pd.read_parquet(RAW / f"{ba}_{kind}.parquet").drop_duplicates("period").set_index("period")["value"].sort_index()
    return d


def regime_masks(df):
    d = df["local_date"]
    hol = df["is_holiday"].astype(bool) | df["day_before_holiday"].astype(bool) | df["day_after_holiday"].astype(bool)
    m = {name: rule(df) for name, rule in evaluate.REGIMES.items()}
    m["Calm (none of the above)"] = ~(m[list(m)[0]] | m[list(m)[1]] | m[list(m)[2]] | hol)
    return m


def write(name, df):
    (OUT / name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"{name}: {len(df):,} rows", flush=True)


# ------------------------------------------------------------------------------------------------
def headline_tables():
    overall, by_year, by_ba, regimes = [], [], [], []
    for v in "AB":
        p = preds(v)
        for name, (drop, eia) in SAMPLES.items():
            c = sample(p, drop, eia)
            ms = M5 + (["eia_df"] if eia else [])
            col = lambda m: "eia_df" if m == "eia_df" else f"pred_{m}"  # noqa: E731
            for m in ms:
                rel = (c[col(m)] - c["y"]) / c["y"]
                overall.append(dict(variant=v, sample=name, model=m, bas=c["ba"].nunique(), hours=len(c),
                                    mape=100 * rel.abs().mean(), peak_hour_mape=100 * rel[c["is_peak"]].abs().mean(),
                                    bias=100 * rel.mean()))
                for y, g in c.groupby(c["local_date"].dt.year):
                    by_year.append(dict(variant=v, sample=name, model=m, year=int(y), hours=len(g), mape=mape(g, col(m))))
                for rg, mask in regime_masks(c).items():
                    g = c[mask.to_numpy()]
                    regimes.append(dict(variant=v, sample=name, regime=rg, model=m, hours=len(g),
                                        mape=mape(g, col(m)) if len(g) else np.nan))
        c = evaluate.common_sample(p)
        w = evaluate.common_sample(p, with_df=True)
        for ba, g in c.groupby("ba"):
            for m in M5:
                by_ba.append(dict(variant=v, ba=ba, model=m, hours=len(g), mape=mape(g, f"pred_{m}")))
        for ba, g in w.groupby("ba"):
            by_ba.append(dict(variant=v, ba=ba, model="eia_df", hours=len(g), mape=mape(g, "eia_df")))
    for name, rows in (("overall.csv", overall), ("by_year.csv", by_year), ("by_ba.csv", by_ba), ("regimes.csv", regimes)):
        write(f"tables/{name}", pd.DataFrame(rows).round(4))


def daily_errors():
    """Per variant, BA and local day: hours and error sums for the 20-BA set and the 17-BA set with an EIA forecast."""
    frames = []
    for v in "AB":
        p = preds(v)
        parts = {}
        for setname, eia in (("all", False), ("eia", True)):
            c = evaluate.common_sample(p, with_df=eia)
            ms = M5 + (["eia_df"] if eia else [])
            hol = (c["is_holiday"].astype(bool) | c["day_before_holiday"].astype(bool) | c["day_after_holiday"].astype(bool))
            cols = {}
            for m in ms:
                col = "eia_df" if m == "eia_df" else f"pred_{m}"
                cols[f"{setname}_{m}_sum_ape"] = ape(c, col)
                cols[f"{setname}_{m}_sum_pe"] = 100 * (c[col] - c["y"]) / c["y"]
                cols[f"{setname}_{m}_sum_ape_peak"] = ape(c, col).where(c["is_peak"], 0.0)
            work = pd.DataFrame(cols, index=c.index)
            work[f"{setname}_hours"] = 1
            work[f"{setname}_peak_hours"] = c["is_peak"].astype(int)
            g = work.groupby([c["ba"], c["local_date"]]).sum()
            if setname == "all":
                g["holiday_adjacent"] = hol.groupby([c["ba"], c["local_date"]]).max()
            parts[setname] = g
        merged = parts["all"].join(parts["eia"], how="left").fillna(0)
        merged["holiday_adjacent"] = merged["holiday_adjacent"].astype(bool)
        merged.insert(0, "variant", v)
        frames.append(merged.reset_index())
    out = pd.concat(frames, ignore_index=True)
    # Sums are kept to 0.1 percentage points as float32: that moves no pooled figure by more than 0.0003
    # points and keeps the file near 4 MB.
    num = out.select_dtypes("float").columns
    out[num] = out[num].round(1).astype("float32")
    for c in [c for c in out.columns if c.endswith("_hours")]:
        out[c] = out[c].astype("int32")
    OUT.mkdir(exist_ok=True)
    out.to_parquet(OUT / "daily_errors.parquet", index=False, compression="zstd", compression_level=19)
    print(f"daily_errors.parquet: {len(out):,} rows, {(OUT / 'daily_errors.parquet').stat().st_size / 1e6:.1f} MB", flush=True)


def extra_models():
    rows = []
    for v in "AB":
        base = preds(v)
        ridge = pd.read_parquet(EVAL / f"extra_{v}_ridge.parquet")[["ba", "stamp", "pred_ridge"]]
        nw = pd.read_parquet(EVAL / "extra_A_lightgbm_noweather.parquet")[["ba", "stamp", "pred_lightgbm_noweather"]]
        data = base.merge(ridge, on=["ba", "stamp"], how="left").merge(nw, on=["ba", "stamp"], how="left")
        for name, eia in (("all_20", False), ("eia_17", True)):
            need = ["pred_ridge", "pred_lightgbm_noweather"]
            c = evaluate.common_sample(data, with_df=eia).dropna(subset=need)
            for m in M5 + ["ridge", "lightgbm_noweather"] + (["eia_df"] if eia else []):
                col = "eia_df" if m == "eia_df" else f"pred_{m}"
                rel = (c[col] - c["y"]) / c["y"]
                rows.append(dict(variant=v, sample=name, model=m, bas=c["ba"].nunique(), hours=len(c), mape=100 * rel.abs().mean(),
                                 peak_hour_mape=100 * rel[c["is_peak"]].abs().mean(), bias=100 * rel.mean()))
            by_year = c.groupby(c["local_date"].dt.year)
            for y, g in by_year:
                for m in ["lightgbm", "ridge", "lightgbm_noweather", "previous_day_naive"] + (["eia_df"] if eia else []):
                    col = "eia_df" if m == "eia_df" else f"pred_{m}"
                    rows.append(dict(variant=v, sample=f"{name}_{int(y)}", model=m, bas=g["ba"].nunique(), hours=len(g),
                                     mape=mape(g, col), peak_hour_mape=np.nan, bias=np.nan))
    write("tables/extra_models.csv", pd.DataFrame(rows).round(4))


def origin_sensitivity():
    base = preds("A")
    base = base[pd.to_datetime(base["origin_date"]) >= "2024-01-01"]
    sets = {6: preds("A_origin6"), 10: base, 12: preds("A_origin12")}
    rows = []
    for eia, name in ((False, "all_20"), (True, "eia_17")):
        keys = None
        for p in sets.values():
            k = set(map(tuple, evaluate.common_sample(p, with_df=eia)[["ba", "stamp"]].to_numpy()))
            keys = k if keys is None else keys & k
        kdf = pd.DataFrame(list(keys), columns=["ba", "stamp"])
        for hour, p in sets.items():
            c = p.merge(kdf, on=["ba", "stamp"])
            for m in ["lightgbm", "previous_day_naive"] + (["eia_df"] if eia else []):
                col = "eia_df" if m == "eia_df" else f"pred_{m}"
                rows.append(dict(origin_hour=hour, sample=name, model=m, hours=len(c), mape=mape(c, col)))
    write("tables/origin_sensitivity.csv", pd.DataFrame(rows).round(4))


def scaling_experiment():
    rows = []
    for v in "AB":
        a, b = preds(v), preds(f"{v}_recent_week")
        m = a.merge(b[["ba", "stamp", "pred_lightgbm"]].rename(columns={"pred_lightgbm": "pred_new"}), on=["ba", "stamp"])
        m = m.dropna(subset=["y", "pred_lightgbm", "pred_new", "pred_previous_day_naive"])
        m["peak"] = m["is_peak"]
        def put(measure, group, cur, new):  # noqa: E306
            rows.append(dict(variant=v, measure=measure, group=group, training_mean_scaling=cur, recent_week_scaling=new))
        put("hours", "all", len(m), len(m))
        put("mape", "all", mape(m, "pred_lightgbm"), mape(m, "pred_new"))
        put("bias", "all", float((100 * (m.pred_lightgbm - m.y) / m.y).mean()), float((100 * (m.pred_new - m.y) / m.y).mean()))
        put("peak_hour_mape", "all", mape(m[m.peak], "pred_lightgbm"), mape(m[m.peak], "pred_new"))
        for rg, mask in regime_masks(m).items():
            g = m[mask.to_numpy()]
            put("mape", rg, mape(g, "pred_lightgbm"), mape(g, "pred_new"))
        for ba, g in m.groupby("ba"):
            put("mape", f"BA {ba}", mape(g, "pred_lightgbm"), mape(g, "pred_new"))
            put("bias", f"BA {ba}", float((100 * (g.pred_lightgbm - g.y) / g.y).mean()), float((100 * (g.pred_new - g.y) / g.y).mean()))
        e = m.dropna(subset=["eia_df"])
        put("mape", "17 BAs with an EIA forecast", mape(e, "pred_lightgbm"), mape(e, "pred_new"))
        put("mape", "EIA forecast on the same 17 BAs", mape(e, "eia_df"), mape(e, "eia_df"))
    write("tables/scaling_experiment.csv", pd.DataFrame(rows).round(4))


def cleaning_thresholds():
    rows = []
    for label, tag in (("0.4/2.0 (used)", "A"), ("0.3/3.0", "A_clean_0.3_3.0"), ("0.5/1.8", "A_clean_0.5_1.8")):
        c = evaluate.common_sample(preds(tag))
        for m in M5:
            rows.append(dict(thresholds=label, model=m, hours=len(c), mape=mape(c, f"pred_{m}")))
    t = pd.DataFrame(rows)
    base = t[t["thresholds"] == "0.4/2.0 (used)"].set_index("model")["mape"]
    t["change_from_used"] = t["mape"] - t["model"].map(base)
    write("tables/cleaning_thresholds.csv", t.round(4))


def error_rise():
    a = evaluate.common_sample(preds("A"), with_df=True).copy()
    a["year"] = a["local_date"].dt.year
    models_ = {"lightgbm": "pred_lightgbm", "previous_day_naive": "pred_previous_day_naive", "eia_df": "eia_df"}
    groups = {"17 BAs": a, "17 BAs without SRP and AZPS": a[~a["ba"].isin(["SRP", "AZPS"])], "SRP": a[a["ba"] == "SRP"]}
    rows = []
    for gname, g in groups.items():
        for y, part in g.groupby("year"):
            for m, col in models_.items():
                e = ape(part, col)
                rows.append(dict(group=gname, year=int(y), model=m, hours=len(part), mean=e.mean(), median=e.median(),
                                 share_over_20_pct=100 * (e > 20).mean()))
    write("tables/error_rise_by_year.csv", pd.DataFrame(rows).round(4))
    rows = []
    for ba, g in a.groupby("ba"):
        for m, col in models_.items():
            by = ape(g, col).groupby(g["year"]).mean()
            rows.append(dict(ba=ba, model=m, error_2024=by.get(2024), error_2026=by.get(2026), change=by.get(2026) - by.get(2024)))
    write("tables/error_rise_by_ba.csv", pd.DataFrame(rows).round(4))
    q = a.assign(quarter=a["local_date"].dt.to_period("Q").astype(str))
    rows = [dict(quarter=k, model=m, mean=mape(g, col), bias=float((100 * (g[col] - g["y"]) / g["y"]).mean()))
            for k, g in q.groupby("quarter") for m, col in models_.items()]
    write("tables/error_rise_by_quarter.csv", pd.DataFrame(rows).round(4))
    s = a[a["ba"] == "SRP"].assign(month=lambda d: d["local_date"].dt.to_period("M").astype(str))
    rows = [dict(month=k, model=m, hours=len(g), mape=mape(g, col)) for k, g in s.groupby("month") for m, col in models_.items()]
    write("tables/srp_monthly_error.csv", pd.DataFrame(rows).round(4))
    # LightGBM against the naive baseline, with and without SRP and AZPS
    r = []
    for gname, g in list(groups.items())[:2]:
        by = g.groupby("year")
        for y, part in by:
            r.append(dict(group=gname, year=int(y), lightgbm_over_previous_day=mape(part, "pred_lightgbm") / mape(part, "pred_previous_day_naive")))
    write("tables/error_rise_ratio.csv", pd.DataFrame(r).round(4))


def srp_flows():
    rows, monthly = [], []
    for ba in ("SRP", "AZPS"):
        x = pd.concat([raw(ba, "D").rename("D"), raw(ba, "NG").rename("NG"), raw(ba, "TI").rename("TI")], axis=1, sort=True).dropna()
        x = x[(x["D"] > 0) & (x["NG"] > 0)]
        day = x.resample("D").mean().dropna()
        yr = day.index.year
        chg = day.diff()
        for y in sorted(set(yr)):
            if y < 2021:
                continue
            m = yr == y
            rows.append(dict(ba=ba, year=int(y),
                             interchange_share_of_net_generation_pct=100 * day.loc[m, "TI"].mean() / day.loc[m, "NG"].mean(),
                             daily_change_sd_demand_pct_of_mean=100 * chg.loc[m, "D"].std() / day.loc[m, "D"].mean(),
                             daily_change_sd_net_generation_pct_of_mean=100 * chg.loc[m, "NG"].std() / day.loc[m, "NG"].mean(),
                             daily_change_sd_interchange_pct_of_net_generation=100 * chg.loc[m, "TI"].std() / day.loc[m, "NG"].mean(),
                             corr_demand_change_interchange_change=chg.loc[m, ["D", "TI"]].corr().iloc[0, 1],
                             corr_demand_change_net_generation_change=chg.loc[m, ["D", "NG"]].corr().iloc[0, 1]))
        gap = (x["D"] - (x["NG"] - x["TI"])).abs().groupby(x.index.tz_localize(None).to_period("M"))
        mean = x.groupby(x.index.tz_localize(None).to_period("M")).mean()
        for k in mean.index[mean.index >= "2025-01"]:
            monthly.append(dict(ba=ba, month=str(k), demand=mean.loc[k, "D"], net_generation=mean.loc[k, "NG"], interchange=mean.loc[k, "TI"],
                                mean_abs_gap_demand_vs_generation_minus_interchange=gap.mean()[k]))
    write("tables/srp_azps_flows_by_year.csv", pd.DataFrame(rows).round(4))
    write("tables/srp_azps_flows_by_month.csv", pd.DataFrame(monthly).round(2))


def load_growth():
    clean = {}
    for ba in BAS:
        c, _ = clean_demand(raw(ba, "D"))
        clean[ba] = c
    growth = []
    for ba, c in clean.items():
        y = c.groupby(c.index.year).mean()
        growth.append(dict(ba=ba, **{f"mean_demand_{k}": y[k] for k in range(2021, 2026)}, growth_2021_to_2025_pct=100 * (y[2025] / y[2021] - 1)))
    write("tables/load_growth.csv", pd.DataFrame(growth).round(2))
    a, b = preds("A"), preds("A_recent_week")
    m = a.merge(b[["ba", "stamp", "pred_lightgbm"]].rename(columns={"pred_lightgbm": "pred_new"}), on=["ba", "stamp"]).dropna(subset=["y", "pred_lightgbm", "pred_new"])
    m["quarter"] = m["local_date"].dt.to_period("Q")
    rows = []
    for ba, c in clean.items():
        for q in sorted(m["quarter"].unique()):
            start = q.start_time.tz_localize("UTC")
            hist = c[c.index < start - pd.Timedelta(days=2)]
            recent = hist[hist.index >= start - pd.Timedelta(days=367)]
            g = m[(m["ba"] == ba) & (m["quarter"] == q)]
            if len(g):
                rows.append(dict(ba=ba, quarter=str(q), recent_demand_over_training_mean=recent.mean() / hist.mean(),
                                 bias_training_mean_scaling=float((100 * (g.pred_lightgbm - g.y) / g.y).mean()),
                                 bias_recent_week_scaling=float((100 * (g.pred_new - g.y) / g.y).mean())))
    t = pd.DataFrame(rows).dropna()
    write("tables/growth_bias_by_ba_quarter.csv", t.round(4))
    t["level_bin"] = pd.qcut(t["recent_demand_over_training_mean"], 5)
    bins = t.groupby("level_bin", observed=True).agg(ba_quarters=("ba", "size"), bias_training_mean_scaling=("bias_training_mean_scaling", "mean"),
                                                      bias_recent_week_scaling=("bias_recent_week_scaling", "mean")).reset_index()
    bins["level_bin"] = bins["level_bin"].astype(str)
    corr = pd.DataFrame([dict(level_bin="correlation with level", ba_quarters=len(t),
                              bias_training_mean_scaling=t["recent_demand_over_training_mean"].corr(t["bias_training_mean_scaling"]),
                              bias_recent_week_scaling=t["recent_demand_over_training_mean"].corr(t["bias_recent_week_scaling"]))])
    write("tables/growth_bias_bins.csv", pd.concat([bins, corr]).round(4))


def selection_and_forecast_checks():
    idx = pd.date_range("2019-01-01 00:00", "2020-12-31 23:00", freq="h", tz="UTC")
    rows = []
    for f in sorted(RAW.glob("*_D.parquet")):
        ba = f.name.split("_")[0]
        d = raw(ba, "D")
        s = d.reindex(idx)
        rows.append(dict(ba=ba, has_data_from_2019_01_01=bool(d.index.min() <= idx[0]), hours=len(idx),
                         missing_or_null_pct=100 * s.isna().mean(), zero_or_negative_pct=100 * (s <= 0).mean(), mean_demand_mw=s[s > 0].mean()))
    t = pd.DataFrame(rows)
    t["passes"] = t["has_data_from_2019_01_01"] & (t["missing_or_null_pct"] <= 2) & (t["zero_or_negative_pct"] <= 2)
    t["rank_among_passing_by_mean_demand"] = t["mean_demand_mw"].where(t["passes"]).rank(ascending=False)
    write("tables/ba_selection.csv", t.sort_values("mean_demand_mw", ascending=False).round(3))
    rows = []
    for ba in BAS:
        d, f = raw(ba, "D"), raw(ba, "DF")
        s = pd.concat([d.rename("d"), f.rename("f"), d.shift(168, freq="h").rename("w")], axis=1, sort=True)
        s = s[(s.index >= "2023-01-01") & (s.index < "2024-01-01") & (s["d"] > 0)]
        e, w = s.dropna(subset=["f"]), s.dropna(subset=["w"])
        rows.append(dict(ba=ba, eia_forecast_mape_2023=100 * ((e.f - e.d).abs() / e.d).mean(), eia_forecast_bias_2023=100 * ((e.f - e.d) / e.d).mean(),
                         seasonal_naive_mape_2023=100 * ((w.w - w.d).abs() / w.d).mean()))
    write("tables/eia_forecast_2023.csv", pd.DataFrame(rows).round(3))
    rows = []
    for ba in BAS:
        d = raw(ba, "D")
        c, bad = clean_demand(d)
        near = np.nan
        if ba in EIA_BAS:
            f = raw(ba, "DF").reindex(d[bad].index)
            rem = d[bad]
            ok = f.notna() & (rem > 0)
            near = int((((rem[ok] - f[ok]).abs() / f[ok]) < 0.25).sum())
        rows.append(dict(ba=ba, hours=len(d), removed_hours=int(bad.sum()), removed_pct=100 * bad.mean(), removed_within_25pct_of_eia_forecast=near))
    write("tables/cleaning_flags.csv", pd.DataFrame(rows).round(4))


def shap_table():
    groups = {"Recent demand and lags": models.LEVEL_COLS, "Calendar and holidays": models.CALENDAR_COLS,
              "Weather known at the forecast time": models.WEATHER_A, "Target-day weather (variant B only)": models.WEATHER_B,
              "Which balancing authority": ["ba"]}
    rows = []
    for v in "AB":
        s = pd.read_csv(EVAL / f"shap_{v}.csv", index_col=0).iloc[:, 0]
        for g, cols in groups.items():
            cols = [c for c in cols if c in s.index]
            if cols:
                rows.append(dict(variant=v, kind="group", name=g, mean_abs_shap_pct_of_ba_demand=s[cols].sum(), share_pct=100 * s[cols].sum() / s.sum()))
        rows += [dict(variant=v, kind="feature", name=f, mean_abs_shap_pct_of_ba_demand=val, share_pct=100 * val / s.sum()) for f, val in s.items()]
    write("tables/shap.csv", pd.DataFrame(rows).round(4))


def weather_checks():
    pairs = ROOT / "data" / "iem_isd_pairs.parquet"
    rows = []
    if pairs.exists():
        p = pd.read_parquet(pairs)
        both = p.dropna(subset=["temp_c", "temp"])
        d = (both["temp_c"] - both["temp"]).abs()
        rows.append(dict(measure="matched temperature readings, Mesonet and ISD", value=len(both)))
        rows.append(dict(measure="mean absolute difference, degrees C", value=d.mean()))
        rows.append(dict(measure="readings differing by more than 0.5 C", value=int((d > 0.5).sum())))
        rows.append(dict(measure="readings differing by more than 2 C", value=int((d > 2).sum())))
        rows.append(dict(measure="rows in the pairs file", value=len(p)))
    write("tables/weather_source_overlap.csv", pd.DataFrame(rows))
    cov = ROOT / "data" / "weather_coverage.csv"
    if cov.exists():
        write("tables/weather_coverage.csv", pd.read_csv(cov))
    wx = pd.read_parquet(ROOT / "data" / "weather_hourly.parquet")
    rows = []
    for ba in ("ERCO", "PJM"):
        d = raw(ba, "D")
        d = d[(d.index >= "2023-07-01") & (d.index < "2023-09-01")]
        t = wx[wx["ba"] == ba].set_index("period")["temp_c"]
        for lag in range(-6, 7):
            j = pd.concat([d.rename("d"), t.shift(lag, freq="h").rename("t")], axis=1, sort=True).dropna()
            rows.append(dict(ba=ba, temperature_shift_hours=lag, hours=len(j), correlation=j["d"].corr(j["t"])))
    write("tables/demand_temperature_correlation.csv", pd.DataFrame(rows).round(4))


if __name__ == "__main__":
    for step in (headline_tables, daily_errors, extra_models, origin_sensitivity, scaling_experiment, cleaning_thresholds,
                 error_rise, srp_flows, load_growth, selection_and_forecast_checks, shap_table, weather_checks):
        print(f"--- {step.__name__}", flush=True)
        step()
