# Electricity Demand Forecasting

Next-day hourly electricity demand for 20 US balancing authorities: LightGBM against simple baselines and the day-ahead forecasts operators report to EIA. Walk-forward validation, quarterly refits, weather variants A and B.

## Commands

```
python -m pytest                        # offline; synthetic data and committed results only
python scripts/run_evaluation.py --variant A    # the walk-forward (needs data/, about 20 to 30 minutes)
python scripts/build_results.py                 # rewrites results/
python scripts/bootstrap_intervals.py           # rewrites results/bootstrap.csv
python scripts/build_dashboard_data.py && python scripts/build_site.py   # rebuilds site/
```

## Rules

- `data/` is git-ignored. Anything the docs quote lives in `results/`. Change a figure by rerunning the pipeline, then `build_results.py`, then update the text from the files; do not type a figure a file computes.
- Pipeline code (`electricity/`, `scripts/`) changes go in their own commits, separate from documentation. Do not edit working pipeline code to fix a documentation issue.
- A forecast uses no input after 08:00 (demand) or 09:00 (weather) local time on the day it is made. The leakage tests in `tests/test_features.py` guard this; keep them passing.
- Test-mutation checks: copy a file to a backup before breaking it and restore from the backup. Never undo with `git checkout -- <file>` when the file has uncommitted real edits.
- Wording: neutral present tense; no status, history or changelog language; associations, never causes; the operator forecast is "reported to EIA", not EIA's own.
- The API key comes from the `EIA_API_KEY` environment variable and never appears in code, docs or output.
