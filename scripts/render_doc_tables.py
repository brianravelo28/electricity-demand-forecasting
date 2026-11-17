"""Write the tables in README.md and docs/METHODOLOGY.md from the committed results files.

    python scripts/render_doc_tables.py          # rewrite the tables in place
    python scripts/render_doc_tables.py --check  # exit 1 if a table in the docs differs from its files

A table sits between `<!-- table:NAME -->` and `<!-- /table -->`. Everything between the markers is
generated from results/, so a figure in a table cannot drift from its file; tests/test_docs.py runs
the check. Prose figures are covered by the claims in tests/test_docs.py.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
DOCS = [ROOT / "README.md", ROOT / "docs" / "METHODOLOGY.md"]

LABEL = {
    "seasonal_naive": "Same hour last week", "previous_day_naive": "Same hour yesterday",
    "same_day_avg": "Average of the last four same weekdays", "regression": "Regression (hour of week, holidays, temperature)",
    "ridge": "Ridge regression, same inputs as LightGBM", "lightgbm_noweather": "LightGBM without weather",
    "lightgbm": "LightGBM", "eia_df": "Operator forecast reported to EIA",
}


def table(headers, rows, align=None):
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def f(v, d=2):
    return "n/a" if pd.isna(v) else f"{v:.{d}f}"


def read(name):
    return pd.read_csv(TABLES / name)


def headline():
    t = read("overall.csv")
    def val(variant, sample, model):
        r = t[(t.variant == variant) & (t["sample"] == sample) & (t.model == model)]
        return f(r["mape"].iloc[0]) if len(r) else "n/a"
    rows = [[LABEL[m], val("A", "all_20", m), val("B", "all_20", m), val("A", "eia_17", m), val("B", "eia_17", m)]
            for m in ["seasonal_naive", "previous_day_naive", "same_day_avg", "regression", "lightgbm", "eia_df"]]
    return table(["Forecast", "A, 20 BAs", "B, 20 BAs", "A, 17 BAs", "B, 17 BAs"], rows)


def extra_models():
    t = read("extra_models.csv")
    t = t[~t["sample"].str.contains(r"_\d{4}$")]
    def val(variant, model, col="mape"):
        r = t[(t.variant == variant) & (t["sample"] == "all_20") & (t.model == model)]
        return f(r[col].iloc[0])
    rows = [[LABEL[m], val("A", m), val("B", m), val("A", m, "peak_hour_mape"), val("B", m, "peak_hour_mape")]
            for m in ["previous_day_naive", "regression", "ridge", "lightgbm_noweather", "lightgbm"]]
    hours = int(t[(t.variant == "A") & (t["sample"] == "all_20")]["hours"].iloc[0])
    return table(["Forecast", "Error, A", "Error, B", "Peak-hour error, A", "Peak-hour error, B"], rows) + \
        f"\n\nAll rows: the {hours:,} hours that all seven forecasts cover (variant A)."


def by_year():
    t = read("by_year.csv")
    t = t[t["sample"] == "eia_17"]
    def val(variant, model, year):
        return f(t[(t.variant == variant) & (t.model == model) & (t.year == year)]["mape"].iloc[0])
    rows = [[y, val("A", "previous_day_naive", y), val("A", "lightgbm", y), val("B", "lightgbm", y), val("A", "eia_df", y)]
            for y in range(2021, 2027)]
    return table(["Year", "Same hour yesterday", "LightGBM, A", "LightGBM, B", "Operator forecast"], rows)


def regimes():
    t = read("regimes.csv")
    t = t[t["sample"] == "eia_17"]
    names = list(dict.fromkeys(t["regime"]))
    def val(variant, model, regime):
        return f(t[(t.variant == variant) & (t.model == model) & (t.regime == regime)]["mape"].iloc[0])
    rows = []
    for rg in names:
        hours = int(t[(t.variant == "A") & (t.model == "lightgbm") & (t.regime == rg)]["hours"].iloc[0])
        rows.append([rg, f"{hours:,}", val("A", "previous_day_naive", rg), val("A", "lightgbm", rg), val("B", "lightgbm", rg), val("A", "eia_df", rg)])
    return table(["Regime", "Hours", "Same hour yesterday", "LightGBM, A", "LightGBM, B", "Operator forecast"], rows)


def interval(r):
    return f"{r['difference_points']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"


def bootstrap_pooled():
    b = pd.read_csv(RESULTS / "bootstrap.csv")
    b = b[b["period"] == "2021 to 2026"]
    rows = []
    for comp, sample in dict.fromkeys(zip(b["comparison"], b["sample"])):
        a = b[(b.comparison == comp) & (b["sample"] == sample) & (b.variant == "A")].iloc[0]
        bb = b[(b.comparison == comp) & (b["sample"] == sample) & (b.variant == "B")].iloc[0]
        rows.append([comp.replace("EIA forecast", "operator forecast"), sample, interval(a), interval(bb)])
    return table(["Comparison (first minus second)", "Hours", "Variant A", "Variant B"], rows, ["l", "l", "r", "r"])


def bootstrap_eia_by_year():
    b = pd.read_csv(RESULTS / "bootstrap.csv")
    b = b[(b.comparison == "LightGBM minus EIA forecast") & (b["sample"] == "17 BAs with an EIA forecast")]
    rows = [[p, interval(b[(b.period == p) & (b.variant == "A")].iloc[0]), interval(b[(b.period == p) & (b.variant == "B")].iloc[0])]
            for p in ["2021", "2022", "2023", "2024", "2025", "2026", "2021 to 2026"]]
    return table(["Period", "Variant A", "Variant B"], rows)


def funnel():
    t = read("funnel.csv")
    t = t[(t.variant == "A") & ~t.stage.str.startswith("  ")]
    return table(["Stage (variant A)", "Hours"], [[r.stage, f"{r.hours:,}"] for r in t.itertuples()])


def origin():
    t = read("origin_sensitivity.csv")
    rows = []
    for h in (6, 10, 12):
        def val(sample, model):
            return f(t[(t.origin_hour == h) & (t["sample"] == sample) & (t.model == model)]["mape"].iloc[0])
        rows.append([f"{h:02d}:00", val("all_20", "lightgbm"), val("all_20", "previous_day_naive"), val("eia_17", "lightgbm"),
                     val("eia_17", "previous_day_naive"), val("eia_17", "eia_df")])
    return table(["Origin", "LightGBM, 20 BAs", "Yesterday, 20 BAs", "LightGBM, 17 BAs", "Yesterday, 17 BAs", "Operator forecast, 17 BAs"], rows)


def thresholds():
    t = read("cleaning_thresholds.csv")
    rows = []
    for th in dict.fromkeys(t["thresholds"]):
        s = t[t.thresholds == th].set_index("model")
        rows.append([th, f"{int(s['hours'].iloc[0]):,}", f(s.loc["previous_day_naive", "mape"], 3), f(s.loc["regression", "mape"], 3), f(s.loc["lightgbm", "mape"], 3)])
    return table(["Thresholds", "Hours", "Yesterday", "Regression", "LightGBM"], rows)


def scaling_summary():
    t = read("scaling_experiment.csv")
    rows = []
    for measure, group in (("mape", "all"), ("bias", "all"), ("peak_hour_mape", "all"), ("mape", "Winter Storm Uri (2021-02-13 to 02-19)"),
                           ("mape", "Winter Storm Elliott (2022-12-21 to 12-27)"), ("mape", "Texas heat, ERCOT (2023-06-20 to 09-15)"),
                           ("mape", "Holidays and adjacent days"), ("mape", "Calm (none of the above)")):
        r = {v: t[(t.variant == v) & (t.measure == measure) & (t.group == group)].iloc[0] for v in "AB"}
        label = {"mape": "Error", "bias": "Bias", "peak_hour_mape": "Peak-hour error"}[measure] + ("" if group == "all" else f", {group.split(' (')[0]}")
        rows.append([label, f(r["A"].training_mean_scaling), f(r["A"].recent_week_scaling), f(r["B"].training_mean_scaling), f(r["B"].recent_week_scaling)])
    return table(["Measure", "A, training mean", "A, previous week", "B, training mean", "B, previous week"], rows)


def shap_groups():
    t = read("shap.csv")
    t = t[t.kind == "group"]
    names = list(dict.fromkeys(t["name"]))
    def val(v, n):
        r = t[(t.variant == v) & (t.name == n)]
        return "n/a" if r.empty else f"{r['share_pct'].iloc[0]:.0f}%"
    return table(["Input group", "Variant A", "Variant B"], [[n, val("A", n), val("B", n)] for n in names])


def selection():
    t = read("ba_selection.csv")
    t = t[t["passes"]].sort_values("mean_demand_mw", ascending=False).head(20)
    rows = [[int(i + 1), r.ba, r.name, f"{r.mean_demand_mw:,.0f}", f(r.missing_or_null_pct)] for i, r in enumerate(t.itertuples())]
    return table(["Rank", "Code", "Name (EIA)", "Mean demand, MW", "Missing or null, %"], rows, ["r", "l", "l", "r", "r"])


def ba_funnel():
    t = read("ba_funnel.csv").fillna("")
    rows = [[r.stage + (f" ({r.codes})" if r.codes else ""), r.bas] for r in t.itertuples()]
    return table(["Stage", "BAs"], rows, ["l", "r"])


def srp_flows():
    t = read("srp_azps_flows_by_year.csv")
    rows = [[r.ba, r.year, f"{r.interchange_share_of_net_generation_pct:.1f}", f"{r.daily_change_sd_demand_pct_of_mean:.1f}",
             f"{r.daily_change_sd_net_generation_pct_of_mean:.1f}", f"{r.corr_demand_change_interchange_change:+.2f}"] for r in t.itertuples()]
    return table(["BA", "Year", "Interchange, % of net generation", "Demand daily change, sd % of mean", "Net generation daily change, sd % of mean",
                  "Correlation of demand and interchange changes"], rows, ["l", "r", "r", "r", "r", "r"])


def error_rise():
    t = read("error_rise_by_year.csv")
    rows = []
    for g in ["17 BAs", "17 BAs without SRP and AZPS", "SRP"]:
        for y in (2024, 2025, 2026):
            s = t[(t.group == g) & (t.year == y)].set_index("model")
            rows.append([g, y, f(s.loc["lightgbm", "mean"]), f(s.loc["previous_day_naive", "mean"]), f(s.loc["eia_df", "mean"]),
                         f(s.loc["lightgbm", "median"]), f(s.loc["eia_df", "median"]),
                         f"{s.loc['lightgbm', 'share_over_20_pct']:.1f}", f"{s.loc['eia_df', 'share_over_20_pct']:.1f}"])
    return table(["Group", "Year", "LightGBM", "Yesterday", "Operator forecast", "LightGBM median", "Operator median",
                  "LightGBM, % of hours over 20", "Operator, % of hours over 20"], rows, ["l"] + ["r"] * 8)


BLOCKS = {fn.__name__: fn for fn in (headline, extra_models, by_year, regimes, bootstrap_pooled, bootstrap_eia_by_year, funnel, origin,
                                      thresholds, scaling_summary, shap_groups, selection, ba_funnel, srp_flows, error_rise)}
# A block is an open marker line, the table, and a close marker line. The table may be empty. The pattern
# must stop at the first close marker: a block never reaches into the prose after it.
PATTERN = re.compile(r"(<!-- table:(\w+) -->\n)(.*?)(<!-- /table -->)", re.S)


def render(text):
    """The text with every generated table rewritten from the results files."""
    def swap(m):
        name = m.group(2)
        if name not in BLOCKS:
            raise SystemExit(f"unknown table block: {name}")
        if "<!-- table:" in m.group(3):
            raise SystemExit(f"table block {name} contains another open marker; the markers are mismatched")
        return m.group(1) + BLOCKS[name]() + "\n" + m.group(4)
    return PATTERN.sub(swap, text)


def main():
    check = "--check" in sys.argv
    stale = []
    for path in DOCS:
        if not path.exists():
            continue
        old = path.read_text(encoding="utf-8")
        new = render(old)
        if new != old:
            stale.append(path.name)
            if not check:
                path.write_text(new, encoding="utf-8", newline="\n")
    if check and stale:
        raise SystemExit(f"tables out of date in: {', '.join(stale)}")
    print("checked" if check else "written", "; changed:" if stale else "; no changes", ", ".join(stale))


if __name__ == "__main__":
    main()
