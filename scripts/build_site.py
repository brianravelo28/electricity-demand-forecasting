"""Assemble the static site that Render serves (site/), from the dashboard page and its data.

    python scripts/build_site.py

Run scripts/build_dashboard_data.py first. This copies the ten JSON files the dashboard reads from
data/dashboard/ into site/results/, copies dashboard/index.html to site/page.html, and writes
site/index.html from dashboard/preview.html, the small stand-in for the claude.ai dashboard runtime,
with its paths pointed at the site's own files and its "local preview" wording removed.
The site/ folder is committed so Render can deploy it straight from the repository.
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "dashboard"
SITE = ROOT / "site"
USED = ["overall", "by_year", "regimes", "by_ba", "monthly", "hourly", "shap", "windows", "bas", "models"]


def swap(text, old, new):
    assert old in text, f"expected to find {old!r} in dashboard/preview.html"
    return text.replace(old, new)


def main():
    shim = (ROOT / "dashboard" / "preview.html").read_text(encoding="utf-8")
    shim = re.sub(r"<!--.*?-->\s*", "", shim, count=1, flags=re.S)  # the "local preview" instructions
    shim = swap(shim, "<title>Electricity demand forecasts (local preview)</title>", "<title>Electricity demand forecasts</title>")
    shim = swap(shim, '<meta name="viewport" content="width=device-width, initial-scale=1">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                '<meta name="description" content="Next-day hourly electricity demand forecasts for 20 US balancing authorities: '
                'LightGBM against simple baselines and EIA\'s own forecasts.">')
    shim = swap(shim, "fetch('../data/dashboard/' + id + '.json')", "fetch('results/' + id + '.json')")
    shim = swap(shim, "fetch('index.html')", "fetch('page.html')")

    (SITE / "results").mkdir(parents=True, exist_ok=True)
    (SITE / "index.html").write_text(shim, encoding="utf-8")
    shutil.copyfile(ROOT / "dashboard" / "index.html", SITE / "page.html")
    total = 0
    for name in USED:
        src = SRC / f"{name}.json"
        shutil.copyfile(src, SITE / "results" / f"{name}.json")
        total += src.stat().st_size
    print(f"site/index.html, site/page.html and {len(USED)} data files ({total / 1024 / 1024:.1f} MB) written to {SITE}")


if __name__ == "__main__":
    main()
