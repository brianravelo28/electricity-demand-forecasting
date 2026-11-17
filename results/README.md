# Results

Every figure the project quotes is in this folder as a CSV file, and `daily_errors.parquet` holds enough to recompute the headline tables. Nothing here needs the EIA API key or the git-ignored `data/` folder.

Regenerate it from the saved evaluation outputs and raw data (the full pipeline takes several hours):

```
python scripts/build_results.py
python scripts/bootstrap_intervals.py
```

## Conventions

- **Error** is the mean absolute percentage error over hourly forecasts: the average size of the miss as a share of actual demand, in percent. **Bias** is the mean signed percentage error (above 0 means the forecast is too high). **Peak-hour error** is the error at each BA's highest-demand hour of each day. Differences between errors are in percentage points.
- **Variant A** uses weather observed up to 09:00 on the day the forecast is made. **Variant B** also uses the weather observed on the day being forecast, which no live system has; it is an upper bound.
- **Sample.** `all_20` is every hour where the actual demand and all five main forecasts exist (20 BAs). `eia_17` adds the requirement that the BA's day-ahead forecast to EIA exists, which leaves out FPC, PSCO and LGEE. `_without_SRP` leaves one BA out of the scoring; the models were fit with it.
- **Test period.** Forecast origins from 2021-01-01 to 2026-11-01, refit every quarter on earlier data. 2026 covers January to October.

## Files

| File | What it holds | Built by |
|---|---|---|
| `daily_errors.parquet` | Per variant, BA and local day: hours, peak hours, and sums of absolute percentage error, signed percentage error and peak-hour absolute error for each model, for the `all` and `eia` samples, plus a holiday-adjacent flag. A table is its sum over a group of rows divided by the hours. Sums are kept to 0.1 percentage points (no pooled figure moves by more than 0.0003 points) | `build_results.py` |
| `tables/overall.csv`, `by_year.csv`, `by_ba.csv`, `regimes.csv` | The headline tables for each variant and sample, each recomputable from `daily_errors.parquet` | `build_results.py` |
| `tables/extra_models.csv` | The five main forecasts plus a ridge regression on the same inputs as LightGBM and LightGBM without weather, on the hours all of them cover | `build_results.py` |
| `bootstrap.csv` | The difference in error between pairs of forecasts with a 95% interval from resampling whole weeks (2,000 resamples, seed 20261117), overall and by year | `bootstrap_intervals.py` |
| `tables/origin_sensitivity.csv` | Results at forecast origins 06:00, 10:00 and 12:00, from 2024 onward | `build_results.py` |
| `tables/scaling_experiment.csv` | LightGBM scaled by the training mean (used) and by the previous week's mean | `build_results.py` |
| `tables/cleaning_thresholds.csv` | Results with the cleaning thresholds 0.4/2.0 (used), 0.3/3.0 and 0.5/1.8 | `build_results.py` |
| `tables/error_rise_*.csv`, `srp_monthly_error.csv` | Error by year, BA, quarter and month for the 17-BA sample in variant A | `build_results.py` |
| `tables/srp_azps_flows_*.csv` | Net generation and interchange against reported demand for SRP and AZPS. Days are the mean of the hours where demand, net generation and interchange are all present and positive | `build_results.py` |
| `tables/load_growth.csv`, `growth_bias_*.csv` | Mean demand by year, and LightGBM's bias against how far recent demand sits above the training level | `build_results.py` |
| `tables/ba_selection.csv`, `eia_forecast_2023.csv`, `cleaning_flags.csv` | The BA selection rule on 2019 to 2020 data, the 2023 check of EIA's forecast, and hours removed by the cleaning rule | `build_results.py` |
| `tables/shap.csv` | Mean absolute SHAP value by input and input group | `build_results.py` |
| `tables/weather_*.csv`, `demand_temperature_correlation.csv` | Weather coverage, the Mesonet against NOAA overlap, and the demand-to-temperature correlation by hour shift | `build_results.py` |

The sensitivity, error-rise and diagnostic tables are outputs of the scripts above; only the four headline tables are recomputable from `daily_errors.parquet`.
