"""Checks on the committed results folder and the dashboard files built from it.

They read only committed files, so they run anywhere. The daily error sums must recompute every
headline table, the dashboard data must equal the tables, the bootstrap file must agree with the
tables it summarizes, and every file in results/ must be described in its README.
"""
import fnmatch
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
SITE = ROOT / "site" / "results"

# sample name in the tables: (BA left out, which set of hours in daily_errors.parquet)
SAMPLES = {"all_20": (None, "all"), "eia_17": (None, "eia"),
           "all_19_without_SRP": ("SRP", "all"), "eia_16_without_SRP": ("SRP", "eia")}
SITE_SAMPLE = {"all": "all_20", "eia": "eia_17", "all_no_srp": "all_19_without_SRP", "eia_no_srp": "eia_16_without_SRP"}
MODEL_ORDER = ["seasonal_naive", "previous_day_naive", "same_day_avg", "regression", "lightgbm", "eia_df"]
TOLERANCE = 0.002  # percentage points; the daily sums are rounded to 0.1


@pytest.fixture(scope="module")
def daily():
    d = pd.read_parquet(RESULTS / "daily_errors.parquet")
    d["local_date"] = pd.to_datetime(d["local_date"])
    return d


def rows_for(daily, variant, sample):
    left_out, _ = SAMPLES[sample]
    return daily[(daily["variant"] == variant) & ((daily["ba"] != left_out) if left_out else True)]


def table_value(x, sample, model, kind="ape"):
    _, s = SAMPLES[sample]
    return x[f"{s}_{model}_sum_{kind}"].sum() / x[f"{s}_hours"].sum()


def test_daily_sums_recompute_the_overall_table(daily):
    overall = pd.read_csv(TABLES / "overall.csv")
    assert len(overall) == 44
    for _, r in overall.iterrows():
        x = rows_for(daily, r["variant"], r["sample"])
        s = SAMPLES[r["sample"]][1]
        assert x[f"{s}_hours"].sum() == r["hours"]
        assert table_value(x, r["sample"], r["model"]) == pytest.approx(r["mape"], abs=TOLERANCE)
        assert table_value(x, r["sample"], r["model"], "pe") == pytest.approx(r["bias"], abs=TOLERANCE)
        peak = x[f"{s}_{r['model']}_sum_ape_peak"].sum() / x[f"{s}_peak_hours"].sum()
        assert peak == pytest.approx(r["peak_hour_mape"], abs=TOLERANCE)


def test_daily_sums_recompute_the_by_year_and_by_ba_tables(daily):
    for _, r in pd.read_csv(TABLES / "by_year.csv").iterrows():
        x = rows_for(daily, r["variant"], r["sample"])
        x = x[x["local_date"].dt.year == r["year"]]
        assert x[f"{SAMPLES[r['sample']][1]}_hours"].sum() == r["hours"]
        assert table_value(x, r["sample"], r["model"]) == pytest.approx(r["mape"], abs=TOLERANCE)
    for _, r in pd.read_csv(TABLES / "by_ba.csv").iterrows():
        s = "eia" if r["model"] == "eia_df" else "all"
        x = daily[(daily["variant"] == r["variant"]) & (daily["ba"] == r["ba"])]
        assert x[f"{s}_hours"].sum() == r["hours"]
        assert x[f"{s}_{r['model']}_sum_ape"].sum() / x[f"{s}_hours"].sum() == pytest.approx(r["mape"], abs=TOLERANCE)


def regime_mask(x, regime):
    d, hol = x["local_date"], x["holiday_adjacent"]
    uri, elliott = d.between("2021-02-13", "2021-02-19"), d.between("2022-12-21", "2022-12-27")
    heat = (x["ba"] == "ERCO") & d.between("2023-06-20", "2023-09-15")
    named = {"Winter Storm Uri (2021-02-13 to 02-19)": uri, "Winter Storm Elliott (2022-12-21 to 12-27)": elliott,
             "Texas heat, ERCOT (2023-06-20 to 09-15)": heat, "Holidays and adjacent days": hol}
    return named.get(regime, ~(uri | elliott | heat | hol))


def test_daily_sums_recompute_the_regime_table(daily):
    regimes = pd.read_csv(TABLES / "regimes.csv")
    assert regimes["regime"].nunique() == 5
    for _, r in regimes.iterrows():
        x = rows_for(daily, r["variant"], r["sample"])
        x = x[regime_mask(x, r["regime"])]
        s = SAMPLES[r["sample"]][1]
        assert x[f"{s}_hours"].sum() == r["hours"]
        if r["hours"]:
            assert table_value(x, r["sample"], r["model"]) == pytest.approx(r["mape"], abs=TOLERANCE)


def test_the_eia_hours_are_a_subset_of_the_all_hours(daily):
    assert (daily["eia_hours"] <= daily["all_hours"]).all()
    assert daily.loc[daily["ba"].isin(["FPC", "PSCO", "LGEE"]), "eia_hours"].sum() == 0  # no usable EIA forecast
    assert daily.groupby(["variant", "ba", "local_date"]).size().max() == 1


def test_the_dashboard_data_equals_the_results_tables():
    tables = {k: pd.read_csv(TABLES / f"{k}.csv") for k in ("overall", "by_year", "by_ba", "regimes")}
    for r in json.loads((SITE / "overall.json").read_text()):
        t = tables["overall"]
        row = t[(t["variant"] == r["variant"]) & (t["sample"] == SITE_SAMPLE[r["sample"]]) & (t["model"] == r["model"])]
        assert len(row) == 1
        assert r["mape"] == pytest.approx(row["mape"].iloc[0], abs=0.001) and r["hours"] == row["hours"].iloc[0]
    for r in json.loads((SITE / "by_year.json").read_text()):
        t = tables["by_year"]
        row = t[(t["variant"] == r["variant"]) & (t["sample"] == SITE_SAMPLE[r["sample"]]) & (t["model"] == r["model"]) & (t["year"] == r["year"])]
        assert len(row) == 1 and r["mape"] == pytest.approx(row["mape"].iloc[0], abs=0.001)
    for r in json.loads((SITE / "by_ba.json").read_text()):
        t = tables["by_ba"]
        row = t[(t["variant"] == r["variant"]) & (t["ba"] == r["ba"]) & (t["model"] == r["model"])]
        assert len(row) == 1 and r["mape"] == pytest.approx(row["mape"].iloc[0], abs=0.001)
    for r in json.loads((SITE / "regimes.json").read_text()):
        t = tables["regimes"]
        row = t[(t["variant"] == r["variant"]) & (t["sample"] == SITE_SAMPLE[r["sample"]]) & (t["model"] == r["model"]) & (t["regime"] == r["regime"])]
        assert len(row) == 1 and r["hours"] == row["hours"].iloc[0]


def test_the_bootstrap_file_agrees_with_the_tables_it_summarizes():
    boot = pd.read_csv(RESULTS / "bootstrap.csv")
    overall = pd.read_csv(TABLES / "overall.csv").set_index(["variant", "sample", "model"])
    assert ((boot["ci_low"] <= boot["difference_points"]) & (boot["difference_points"] <= boot["ci_high"])).all()
    assert (boot["interval_excludes_zero"] == ((boot["ci_low"] > 0) | (boot["ci_high"] < 0))).all()
    assert (boot["weeks"] > 0).all() and (boot["hours"] > 0).all()
    pooled = boot[(boot["period"] == "2021 to 2026") & (boot["sample"] == "20 BAs")]
    pairs = {"LightGBM minus yesterday's demand": ("lightgbm", "previous_day_naive"), "LightGBM minus regression": ("lightgbm", "regression")}
    for _, r in pooled.iterrows():
        if r["comparison"] in pairs:
            a, b = pairs[r["comparison"]]
            expected = overall.loc[(r["variant"], "all_20", a), "mape"] - overall.loc[(r["variant"], "all_20", b), "mape"]
            assert r["difference_points"] == pytest.approx(expected, abs=0.01)
    eia = boot[(boot["comparison"] == "LightGBM minus EIA forecast") & (boot["sample"] == "17 BAs with an EIA forecast") & (boot["period"] == "2021 to 2026")]
    for _, r in eia.iterrows():
        expected = overall.loc[(r["variant"], "eia_17", "lightgbm"), "mape"] - overall.loc[(r["variant"], "eia_17", "eia_df"), "mape"]
        assert r["difference_points"] == pytest.approx(expected, abs=0.01)


def test_every_results_file_is_described_in_the_results_readme():
    text = (RESULTS / "README.md").read_text(encoding="utf-8")
    described = set(re.findall(r"`([^`]+\.(?:csv|parquet))`", text))
    files = [p for p in RESULTS.rglob("*") if p.is_file() and p.name != "README.md"]
    assert files
    for p in files:
        rel, name = p.relative_to(RESULTS).as_posix(), p.name
        assert any(fnmatch.fnmatch(name, d.split("/")[-1]) or fnmatch.fnmatch(rel, d) for d in described), f"{rel} is not in results/README.md"
