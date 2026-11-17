"""Checks that README.md and docs/METHODOLOGY.md agree with the committed results.

Tables in the documents are generated (scripts/render_doc_tables.py) and the first test fails when one
differs from its files. Figures typed into the prose are covered one by one: each test builds the
string from a results file and requires it to appear in the document. The last tests cover structure:
the README length, links and anchors, and wording the project rules out.
"""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
README = (ROOT / "README.md").read_text(encoding="utf-8")
METHOD = (ROOT / "docs" / "METHODOLOGY.md").read_text(encoding="utf-8")
BOTH = README + "\n" + METHOD


def csv(name):
    return pd.read_csv(TABLES / name)


def overall(variant, sample, model):
    t = csv("overall.csv")
    return t[(t.variant == variant) & (t["sample"] == sample) & (t.model == model)].iloc[0]


def extra(variant, model):
    t = csv("extra_models.csv")
    return t[(t.variant == variant) & (t["sample"] == "all_20") & (t.model == model)].iloc[0]


def boot(variant, comparison, sample, period):
    t = pd.read_csv(ROOT / "results" / "bootstrap.csv")
    r = t[(t.variant == variant) & (t.comparison == comparison) & (t["sample"] == sample) & (t.period == period)]
    assert len(r) == 1, (variant, comparison, sample, period)
    return r.iloc[0]


def signed(x, d=2):
    return f"{x:+.{d}f}"


def interval(r):
    return f"{signed(r.difference_points)} [{signed(r.ci_low)}, {signed(r.ci_high)}]"


def text_interval(r):
    return f"{signed(r.difference_points)} points, 95% interval {signed(r.ci_low)} to {signed(r.ci_high)}"


def need(doc, *strings):
    missing = [s for s in strings if s not in doc]
    assert not missing, f"not in the document, or different from the results files: {missing}"


# ---- generated tables ---------------------------------------------------------------------------------

def test_every_generated_table_matches_the_results_files():
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "render_doc_tables.py"), "--check"],
                          capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stdout + done.stderr


def test_table_markers_are_paired_and_filled():
    for name, doc in (("README", README), ("METHODOLOGY", METHOD)):
        opens = re.findall(r"<!-- table:\w+ -->", doc)
        assert len(opens) == doc.count("<!-- /table -->"), name
        assert len(re.findall(r"<!-- table:\w+ -->\n\|", doc)) == len(opens), f"{name} has an empty table block"


# ---- README figures -----------------------------------------------------------------------------------

def test_readme_headline_figures():
    a, b = overall("A", "all_20", "lightgbm"), overall("B", "all_20", "lightgbm")
    prev = overall("A", "all_20", "previous_day_naive")
    ridge = extra("A", "ridge")
    eia17, lgbm17 = overall("A", "eia_17", "eia_df"), overall("A", "eia_17", "lightgbm")
    gap26 = boot("A", "LightGBM minus EIA forecast", "17 BAs with an EIA forecast", "2026")
    need(README,
         f"by {a.mape:.2f}% on average (n = {int(a.hours):,} hours across 20 BAs)",
         f"against {prev.mape:.2f}% for repeating yesterday's demand and {ridge.mape:.2f}% for a ridge regression",
         f"({eia17.mape:.2f}% against {lgbm17.mape:.2f}%)",
         f"a gap of {text_interval(gap26)}",
         f"reaches {b.mape:.2f}%")


def test_readme_data_section_counts():
    flags, funnel = csv("cleaning_flags.csv"), csv("funnel.csv")
    f = funnel[funnel.variant == "A"].set_index("stage").hours
    target, scored = int(f.iloc[0]), int(f[[s for s in f.index if s.startswith("Scored")][0]])
    need(README,
         f"{int(flags.removed_hours.sum())} of {int(flags.hours.sum()):,} hours across the 20 BAs",
         f"The test period has {target:,} target hours, of which {scored:,} are scored")


def test_readme_what_can_it_do_hour_counts():
    reg, yr = csv("regimes.csv"), csv("by_year.csv")
    heat = reg[(reg.variant == "A") & (reg["sample"] == "all_20") & reg.regime.str.startswith("Texas") & (reg.model == "lightgbm")].hours.iloc[0]
    h26 = yr[(yr.variant == "A") & (yr["sample"] == "eia_17") & (yr.model == "lightgbm") & (yr.year == 2026)].hours.iloc[0]
    need(README, f"{int(heat):,} hours.)", f"{int(h26):,} hours in 2026")


def test_readme_what_the_data_shows():
    ridge = boot("A", "LightGBM minus ridge", "20 BAs", "2021 to 2026")
    noweather = boot("A", "LightGBM minus LightGBM without weather", "20 BAs", "2021 to 2026")
    b_ = extra("B", "lightgbm")
    a_ = extra("A", "lightgbm")
    eia = boot("A", "LightGBM minus EIA forecast", "17 BAs with an EIA forecast", "2021 to 2026")
    by_year = {y: boot("A", "LightGBM minus EIA forecast", "17 BAs with an EIA forecast", str(y)) for y in (2021, 2025, 2026)}
    need(README,
         f"Against ridge the difference is {signed(ridge.difference_points)} points (95% interval {signed(ridge.ci_low)} to {signed(ridge.ci_high)})",
         f"is worth {-noweather.difference_points:.2f} points to LightGBM",
         f"perfect next-day weather is worth {extra('A', 'lightgbm_noweather').mape - b_.mape:.1f}",
         f"ahead by {eia.difference_points:.2f} points (interval {signed(eia.ci_low)} to {signed(eia.ci_high)})",
         f"from {by_year[2021].difference_points:.2f} points in 2021 to {by_year[2025].difference_points:.2f} in 2025 and {signed(by_year[2026].difference_points)} in 2026")
    assert a_.mape > b_.mape


def test_lightgbm_beats_ridge_in_every_ba_and_every_year():
    ba = csv("extra_models_by_ba.csv")
    for v in "AB":
        s = ba[ba.variant == v]
        assert len(s) == 20 and (s.lightgbm < s.ridge).all() and (s.lightgbm < s.lightgbm_noweather).all()
    by_year = csv("extra_models.csv")
    by_year = by_year[by_year["sample"].str.fullmatch(r"all_20_\d{4}")]
    for (v, y), g in by_year.groupby([by_year.variant, by_year["sample"].str[-4:]]):
        m = g.set_index("model").mape
        assert m.lightgbm < m.ridge and m.lightgbm == m.min(), (v, y)
    need(BOTH, "all 20 BAs", "every year")


def test_operator_forecast_leads_each_year_to_2025_and_2026_includes_zero():
    for y in range(2021, 2026):
        assert boot("A", "LightGBM minus EIA forecast", "17 BAs with an EIA forecast", str(y)).ci_low > 0
    r26 = boot("A", "LightGBM minus EIA forecast", "17 BAs with an EIA forecast", "2026")
    assert r26.ci_low < 0 < r26.ci_high
    b = boot("B", "LightGBM minus EIA forecast", "17 BAs with an EIA forecast", "2021 to 2026")
    assert b.ci_high < 0


def test_the_three_bas_without_an_operator_forecast():
    by_ba = csv("by_ba.csv")
    have = set(by_ba[by_ba.model == "eia_df"].ba)
    assert len(have) == 17 and set(by_ba.ba) - have == {"FPC", "PSCO", "LGEE"}
    need(README, "(FPC, PSCO, LGEE)")
    assert csv("weather_coverage.csv").shape[0] == 42
    need(README, "42 airport stations")


# ---- METHODOLOGY figures ------------------------------------------------------------------------------

def test_cleaning_figures():
    flags = csv("cleaning_flags.csv")
    assert flags.removed_during_uri.sum() == 0 and flags.removed_during_elliott.sum() == 3
    need(METHOD,
         f"It removes {int(flags.removed_hours.sum())} of {int(flags.hours.sum()):,} hours",
         f"never more than {flags.removed_pct.max() / 100:.2%} of a BA's hours",
         f"(the largest share is {flags.sort_values('removed_pct').iloc[-1].ba} at {flags.removed_pct.max():.3f}%)",
         "removes nothing during Uri and three hours during Elliott")
    eia_bas = set(csv("by_ba.csv").query("model == 'eia_df'").ba)
    removed17 = int(flags[flags.ba.isin(eia_bas)].removed_hours.sum())
    near = int(flags.removed_within_25pct_of_eia_forecast.sum())
    need(METHOD, f"{near} of the {removed17} hours removed on the 17 BAs")
    out = csv("residual_outliers.csv")
    need(METHOD, f"After cleaning, {int(out[out.variant == 'A'].previous_day_error_over_100pct.iloc[0])} of the scored hours")


def test_cleaning_thresholds_move_no_error_by_more_than_the_stated_amount():
    t = csv("cleaning_thresholds.csv")
    assert set(t.thresholds) == {"0.4/2.0 (used)", "0.3/3.0", "0.5/1.8"}
    assert t.change_from_used.abs().max() <= 0.03
    need(METHOD, "change no model's error by more than 0.03 points")


def test_weather_figures():
    cov, by_ba = csv("weather_coverage.csv"), csv("weather_coverage_by_ba.csv")
    ov = csv("weather_source_overlap.csv").set_index("measure").value
    assert cov.hour_coverage.min() >= 0.993 and by_ba.share_with_temperature_pct.min() >= 99.46
    need(METHOD, "at least 99.3% of hours and every BA in at least 99.46%",
         f"{int(ov['matched temperature readings, Mesonet and ISD']):,} temperature readings match",
         f"by {ov['mean absolute difference, degrees C']:.2f} °C on average",
         f"{int(ov['readings differing by more than 0.5 C'])} differ by more than 0.5 °C and {int(ov['readings differing by more than 2 C'])} by more than 2 °C")
    corr = csv("demand_temperature_correlation.csv")
    for ba, shown in (("ERCO", "0.97"), ("PJM", "0.92")):
        g = corr[corr.ba == ba]
        best = g.loc[g.correlation.idxmax()]
        assert best.temperature_shift_hours == 1  # the shift moves temperature later, so +1 pairs demand with the previous hour
        assert f"{best.correlation:.2f}" == shown, ba
    need(METHOD, "correlates 0.97 with the temperature of the previous hour, and PJM 0.92")


def test_operator_forecast_exclusion_figures():
    t = csv("eia_forecast_2023.csv").set_index("ba")
    need(METHOD,
         f"FPC's is {t.loc['FPC', 'eia_forecast_mape_2023']:.1f}% off and biased low by the same amount",
         f"PSCO's is {t.loc['PSCO', 'eia_forecast_mape_2023']:.1f}% off (biased high by {t.loc['PSCO', 'eia_forecast_bias_2023']:.1f}%)",
         f"LGEE's is {t.loc['LGEE', 'eia_forecast_mape_2023']:.1f}% off (biased low by {-t.loc['LGEE', 'eia_forecast_bias_2023']:.1f}%)")
    assert abs(t.loc["FPC", "eia_forecast_mape_2023"] + t.loc["FPC", "eia_forecast_bias_2023"]) < 0.1
    others = t.drop(["FPC", "PSCO", "LGEE"]).eia_forecast_mape_2023
    need(METHOD, f"against {others.min():.1f}% to {others.max():.1f}% for the other {len(others)}")


def test_regime_overlap_and_yesterday_beating_lightgbm_in_the_heat():
    d = pd.read_parquet(ROOT / "results" / "daily_errors.parquet")
    d = d[d.variant == "A"]
    day = pd.to_datetime(d.local_date)
    uri = (day >= "2021-02-13") & (day <= "2021-02-19")
    both = int(d.loc[uri & d.holiday_adjacent, "eia_hours"].sum())
    need(METHOD, f"{both:,} Uri hours also count as holiday-adjacent")
    reg = csv("regimes.csv")
    heat = reg[(reg.variant == "A") & (reg["sample"] == "all_20") & reg.regime.str.startswith("Texas")].set_index("model").mape
    assert heat.previous_day_naive < heat.lightgbm


def test_ridge_share_of_the_way_and_weather_value():
    prev, ridge, lgbm = (extra("A", m).mape for m in ("previous_day_naive", "ridge", "lightgbm"))
    share = (prev - ridge) / (prev - lgbm)
    need(METHOD, f"gets about {share:.0%} of the way")
    nw = boot("A", "LightGBM minus LightGBM without weather", "20 BAs", "2021 to 2026")
    perfect = extra("A", "lightgbm_noweather").mape - extra("B", "lightgbm").mape
    need(METHOD, f"worth {-nw.difference_points:.2f} points to LightGBM and perfect target-day weather {perfect:.1f} points")


def test_without_srp_comparisons_keep_rankings_and_move_less_than_the_stated_amount():
    o = csv("overall.csv")
    for v in "AB":
        for full, short in (("all_20", "all_19_without_SRP"), ("eia_17", "eia_16_without_SRP")):
            x = o[(o.variant == v) & (o["sample"] == full)].set_index("model").mape
            y = o[(o.variant == v) & (o["sample"] == short)].set_index("model").mape
            assert (y - x).abs().max() <= 0.2, (v, full)
            assert list(x.sort_values().index) == list(y.sort_values().index), (v, full)
    need(METHOD, "no pooled error moves by more than 0.2 points")


def test_srp_may_2026_figures():
    d = csv("srp_may_2026_daily.csv").set_index("date")
    r = lambda day, col: f"{d.loc[day, col]:,.0f}"
    need(METHOD,
         f"net generation stayed between {d.loc['2026-05-03':'2026-05-05', 'net_generation'].min():,.0f} and {d.loc['2026-05-03':'2026-05-05', 'net_generation'].max():,.0f} MW a day",
         f"net interchange rose from {r('2026-05-03', 'interchange')} to {r('2026-05-05', 'interchange')} MW",
         f"reported demand fell from {r('2026-05-03', 'demand_actual')} to {r('2026-05-05', 'demand_actual')} MW",
         f"the operator forecast expected {r('2026-05-04', 'operator_forecast')} and {r('2026-05-05', 'operator_forecast')} MW and LightGBM {r('2026-05-04', 'lightgbm_forecast')} and {r('2026-05-05', 'lightgbm_forecast')} MW")


def test_srp_flow_statements():
    f = csv("srp_azps_flows_by_year.csv").set_index(["ba", "year"])
    assert f.loc[("SRP", 2026), "corr_demand_change_interchange_change"] < -0.5
    assert f.loc[("SRP", 2026), "corr_demand_change_interchange_change"] < f.loc[[("SRP", y) for y in range(2021, 2026)], "corr_demand_change_interchange_change"].min()
    share = f.loc[("SRP", 2024), "interchange_share_of_net_generation_pct"]
    assert 1 / (1 - share / 100) - 1 > 0.65  # generates about 70% more than the demand it reports
    need(METHOD, "about 70% more than the demand it reports")


def test_error_rise_statements():
    r = csv("error_rise_by_ba.csv").pivot(index="ba", columns="model", values="change")
    worse = set(r.index[r.eia_df > r.previous_day_naive + 0.5])
    assert worse == {"SRP", "SWPP", "CPLE", "AZPS", "CISO", "PACE", "NEVP"}, worse
    need(METHOD, "in SRP and in six other BAs: SWPP, CPLE, AZPS, CISO, PACE and NEVP")
    n = csv("newest_days.csv")
    over = n[n.ratio > 1].sort_values("ratio", ascending=False)
    assert list(over.ba) == ["SRP", "FPL", "BPAT", "PACE"]
    need(METHOD, f"the median ratio of the two errors is {n.ratio.median():.2f}", f"SRP most ({over.ratio.iloc[0]:.2f} times")


def test_growth_statements():
    g = csv("load_growth.csv").set_index("ba").growth_2021_to_2025_pct
    need(METHOD, *(f"{ba} up {g[ba]:.0f}% from 2021 to 2025" if ba == "ERCO" else f"{ba} {g[ba]:.0f}%" for ba in ("ERCO", "AZPS", "SRP", "FPL")))
    q = csv("growth_bias_by_ba_quarter.csv")
    c = q.recent_demand_over_training_mean.corr(q.bias_training_mean_scaling)
    need(METHOD, f"Across {len(q)} BA-quarters", f"a correlation of {c:.2f}")


def test_origin_sensitivity_statement():
    o = csv("origin_sensitivity.csv")
    m = o[(o["sample"] == "all_20") & (o.model == "lightgbm")].set_index("origin_hour").mape
    assert list(m.index) == [6, 10, 12]
    need(METHOD, f"changes LightGBM's error by {m[6] - m[10]:.2f} and {m[10] - m[12]:.2f} points")


def test_scaling_statement():
    s = csv("scaling_experiment.csv")
    for v in "AB":
        x = s[s.variant == v].set_index(["measure", "group"])
        assert abs(x.loc[("mape", "all"), "training_mean_scaling"] - x.loc[("mape", "all"), "recent_week_scaling"]) <= 0.03
        bas = x.loc["mape"].loc[[g for g in x.loc["mape"].index if g.startswith("BA ")]]
        better = int((bas.recent_week_scaling < bas.training_mean_scaling).sum())
        assert len(bas) == 20 and 8 <= better <= 12, (v, better)
        assert x.loc[("bias", "all"), "recent_week_scaling"] > 0 > x.loc[("bias", "all"), "training_mean_scaling"]
    for regime in ("Winter Storm Uri (2021-02-13 to 02-19)", "Winter Storm Elliott (2022-12-21 to 12-27)"):
        a = s[(s.variant == "A") & (s.group == regime) & (s.measure == "mape")].iloc[0]
        b = s[(s.variant == "B") & (s.group == regime) & (s.measure == "mape")].iloc[0]
        assert (a.recent_week_scaling < a.training_mean_scaling) and (b.recent_week_scaling > b.training_mean_scaling)
    bins = csv("growth_bias_bins.csv").set_index("level_bin")
    A, B = s[(s.variant == "A") & (s.measure == "bias") & (s.group == "all")].iloc[0], s[(s.variant == "B") & (s.measure == "bias") & (s.group == "all")].iloc[0]
    corr = bins.loc["correlation with level"]
    need(METHOD,
         f"falls from {corr.bias_training_mean_scaling:.2f} to {corr.bias_recent_week_scaling:.2f} across the {int(corr.ba_quarters)} BA-quarters",
         f"from {A.training_mean_scaling:.2f} to {signed(A.recent_week_scaling)} in variant A and from {B.training_mean_scaling:.2f} to {signed(B.recent_week_scaling)} in variant B",
         "Overall error is unchanged in both variants, the scaling helps about half the BAs, and its effect in Uri and Elliott changes sign between variants")


def test_selection_counts():
    resp = csv("ba_funnel.csv").set_index("stage").bas
    assert resp["Respondents on the API route"] == 83 and resp["Used"] == 20
    assert resp["Regional aggregates"] + resp["BAs"] == resp["Respondents on the API route"]
    assert resp["BAs"] - resp["Generation-only BAs with no demand series"] == resp["BAs with a demand series"]
    assert resp["Fail the rule"] + resp["Pass the rule"] == resp["BAs with a demand series"]
    assert resp["Smaller than the 20 largest passing BAs"] + resp["Used"] == resp["Pass the rule"]
    sel = csv("ba_selection.csv")
    assert len(sel) == resp["BAs with a demand series"] and sel.passes.sum() == resp["Pass the rule"]


def test_the_manifest_covers_every_raw_input():
    m = csv("data_manifest.csv")
    assert m.sha256.str.fullmatch(r"[0-9a-f]{64}").all() and len(m) >= 40 and m.file.is_unique
    assert (m.rows > 0).all()


# ---- structure ----------------------------------------------------------------------------------------

def slug(heading):
    s = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return re.sub(r"\s", "-", s)


def anchors(doc):
    return {slug(h) for h in re.findall(r"^#{1,6}\s+(.*)$", doc, flags=re.M)}


def test_the_readme_is_the_stated_length():
    prose = re.sub(r"```.*?```", "", README, flags=re.S)
    prose = re.sub(r"<!-- table:\w+ -->.*?<!-- /table -->", "", prose, flags=re.S)
    words = len(prose.split())
    assert 1000 <= words <= 1400, words


@pytest.mark.parametrize("name,doc,base", [("README.md", README, ROOT), ("METHODOLOGY.md", METHOD, ROOT / "docs")])
def test_local_links_and_anchors_resolve(name, doc, base):
    for target in re.findall(r"\]\(([^)\s]+)\)", doc):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, frag = target.partition("#")
        dest = (base / path) if path else None
        if dest is not None:
            assert dest.exists(), f"{name}: broken link {target}"
        text = (dest.read_text(encoding="utf-8") if dest is not None and dest.suffix == ".md" else doc if not path else None)
        if frag and text is not None:
            assert frag in anchors(text), f"{name}: no heading for #{frag} in {path or name}"


def test_no_status_history_or_causal_wording():
    banned = r"\b(now|currently|recently|previously|no longer|newly|updated|used to|formerly|originally|todo|wip|still|already|changelog)\b"
    hits = re.findall(banned, BOTH, flags=re.I)
    assert not hits, hits
    assert not re.search(r"\b(EIA'?s? (own )?forecast|EIA forecasts)\b", README), "the operator forecast is reported to EIA, not EIA's own"


def test_the_documents_name_no_private_value():
    assert not re.search(r"EIA_API_KEY\s*=\s*\S", BOTH)
    assert not re.search(r"[A-Za-z0-9]{40}", re.sub(r"[0-9a-f]{64}", "", BOTH))
