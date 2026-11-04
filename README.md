# Electricity Demand Forecasting

Tests whether a gradient-boosted model forecasts next-day hourly electricity demand better than naive and seasonal-naive baselines, across US balancing authorities.

Data: U.S. Energy Information Administration, Hourly Electric Grid Monitor (EIA-930), via the EIA API. EIA is the source of the raw demand values; forecasts and metrics are computed by the author.

## Setup

Python 3.14. The API key is read from the `EIA_API_KEY` environment variable (see `.env.example`); a local `.env` file also works.

```
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\test_eia.py
```

## Running the evaluation

Raw data comes from the ingestion scripts in `scripts/` (see `docs/PLANNING.md`). The walk-forward evaluation then runs on saved files only and writes to `data/eval/`:

```
.venv\Scripts\python scripts\run_evaluation.py --variant A
.venv\Scripts\python scripts\run_evaluation.py --variant B
```

Variant A uses weather observed up to the forecast time (09:00 on the day the forecast is made). Variant B also uses the weather observed on the target day, the day being forecast, and is an upper bound, not something a live system has. For a forecast made Tuesday at 10:00 for Wednesday, A sees weather through Tuesday 09:00 and B also sees Wednesday's actual weather. Each run takes about 20 to 30 minutes.

## Dashboard

A single page for readers: the headline results, error by year, by balancing authority and in storms and holidays, a viewer for any BA's forecasts against actual demand in five chosen weeks, and what the model relies on. It reads only saved files, never a live service, and labels what is EIA's and what is computed.

```
.venv\Scripts\python scripts\build_dashboard_data.py
```

To view it locally, serve the project folder and open the preview page. `dashboard/preview.html` stands in for the claude.ai runtime that the published page needs (it supplies the small data and chart API the page uses, and loads d3 from a CDN, so it needs a network connection):

```
.venv\Scripts\python -m http.server 8765
```

Then open `http://localhost:8765/dashboard/preview.html`.

The build script writes small JSON files to `data/dashboard/` from the saved evaluation outputs (run the evaluation and `scripts/run_shap.py` first). The page is `dashboard/index.html`. It was published as a claude.ai dashboard whose data sources are those files, so a change to a figure means rebuilding the files and replacing them there.

### Deploying to Render

`render.yaml` describes a Render static site that serves the `site/` folder, which holds the page and its data pre-built, so Render has nothing to build or run. After changing a figure, rebuild and push:

```
.venv\Scripts\python scripts\build_dashboard_data.py
.venv\Scripts\python scripts\build_site.py
git add site && git commit -m "Rebuild the site" && git push
```

In Render, choose New, then Blueprint, and select this repository (for a private repository, give Render access to it in its GitHub settings). The published site is public to anyone with the link. The charts load d3 from a CDN, so visitors need a network connection. `site/results/` holds computed results only; no raw EIA files are in the repository.

## Data cleaning

EIA's hourly demand contains impossible values (for example about 2.1 billion MWh for PJM). Raw files are kept untouched. A reading is set to missing when it is not positive or falls outside 0.4 to 2.0 times the median of the previous 15 days, using only earlier data. These thresholds were chosen by eye on the full dataset and not fit on training data; rerunning with 0.5/1.8 and 0.3/3.0 changed no overall error by more than 0.05 percentage points. The rule discards about 15 genuine extreme-event hours and leaves a few dozen isolated dips. Details and checks are in `docs/PLANNING.md`.

## Tests

```
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

The tests run on synthetic series and need no downloaded data.
