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

## Tests

```
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

The tests run on synthetic series and need no downloaded data.
