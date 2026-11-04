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

Variant A uses weather observed up to the forecast time. Variant B uses observed weather for the target day and is an upper bound, not something a live system has. Each run takes about 20 to 30 minutes.

## Data cleaning

EIA's hourly demand contains impossible values (for example about 2.1 billion MWh for PJM). Raw files are kept untouched. A reading is set to missing when it is not positive or falls outside 0.4 to 2.0 times the median of the previous 15 days, using only earlier data. These thresholds were chosen by eye on the full dataset and not fit on training data; rerunning with 0.5/1.8 and 0.3/3.0 changed no overall error by more than 0.05 percentage points. The rule discards about 15 genuine extreme-event hours and leaves a few dozen isolated dips. Details and checks are in `docs/PLANNING.md`.

## Tests

```
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

The tests run on synthetic series and need no downloaded data.
