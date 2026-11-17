# Methodology

Contents: [Terms](#terms) · [Data sources](#data-sources) · [Data and cleaning](#data-and-cleaning) · [Forecast design](#forecast-design) · [Models](#models) · [Validation](#validation) · [Results](#results) · [Why error rises in 2025 and 2026](#why-error-rises-in-2025-and-2026) · [Forecast origin](#forecast-origin) · [What the model uses](#what-the-model-uses) · [Scaling by a recent level](#scaling-by-a-recent-level) · [Limitations](#limitations) · [Reproducibility](#reproducibility) · [Deployment notes](#deployment-notes)

Every table below is generated from the files in [results/](../results/README.md) by `scripts/render_doc_tables.py`, and a test fails if a table or a quoted figure differs from those files.

## Terms

| Term | Meaning |
|---|---|
| Balancing authority (BA) | The operator that keeps electricity supply and demand matched in a region. The data has one demand series per BA. |
| EIA-930 | The U.S. Energy Information Administration's Hourly Electric Grid Monitor (Form EIA-930). Each BA reports hourly demand, net generation, interchange with neighbours and a day-ahead demand forecast. |
| Demand | The BA-reported hourly demand in megawatthours, which equals the average megawatts over the hour (EIA series type `D`). |
| Operator forecast | The day-ahead demand forecast a BA reports to EIA, made in the normal course of that BA's business (series type `DF`). EIA publishes it and does not produce it. It is missing or unusable for FPC, PSCO and LGEE. |
| Hour-ending | A timestamp marks the end of its hour: the value stamped 14:00 covers 13:00 to 14:00. EIA's hourly series are on UTC. |
| Forecast origin | 10:00 local time on the day a forecast is made. |
| Target day | The local day after the origin. Its 24 hours (23 or 25 on a clock-change day) are forecast, 14 to 37 hours ahead. |
| Variant A | Weather inputs stop at 09:00 on the day the forecast is made, which is all a live forecaster would have. |
| Variant B | Variant A plus the weather observed on the target day. No live forecaster has it, so variant B is an upper bound on what better weather information could add. |
| Walk-forward | The test period is cut into calendar quarters; each quarter is forecast by models fitted only on earlier origin dates, then the window moves on. |
| Error | Mean absolute percentage error over hourly forecasts: the average size of the miss as a share of actual demand, in percent. Lower is better. |
| Bias | Mean signed percentage error. Above 0 means the forecast is too high. |
| Peak-hour error | The error at each BA's highest-demand hour of each day. |
| Percentage points | The unit of a difference between two errors. |
| Regime | A set of days reported on its own: Winter Storm Uri, Winter Storm Elliott, the 2023 Texas summer heat, holidays and the days next to them, and all other days. |
| Bootstrap interval | A 95% interval from resampling whole calendar weeks, all BAs together, 2,000 times. |
| Ridge regression | A linear model with a penalty on large coefficients, fitted on the same inputs as LightGBM. |
| SHAP | A way of splitting each forecast into contributions from each input. The mean absolute contribution shows what the model uses, not what causes demand. |

## Data sources

| Source | What | Version used |
|---|---|---|
| [EIA API v2](https://www.eia.gov/opendata/), route `electricity/rto/region-data` | Hourly demand (`D`) and day-ahead forecast (`DF`) for every BA; net generation (`NG`) and interchange (`TI`) for SRP and AZPS | Pulled 2026-11-03 (and 2026-11-04 for `NG` and `TI`). History starts 2019-01-01 and ends 2026-11-03 17:00 UTC. The API serves the latest revised values |
| EIA-930 reference tables | The list of respondents (BAs and regional aggregates) and each BA's time zone | As on the download date |
| [NOAA Integrated Surface Database](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database), public S3 bucket `noaa-global-hourly-pds` | Hourly temperature, dew point and wind speed at airport stations | 2019-01-01 to 2025-08-27, where the archive ends |
| [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) airport observations | The same stations after the NOAA archive ends | 2025-08-20 to 2026-11-03 |
| U.S. federal holiday calendar | Holiday flags | The calendar built into pandas |

**Data versions.** EIA revises its series, so a later pull can differ. [results/tables/data_manifest.csv](../results/tables/data_manifest.csv) holds the row count and SHA-256 of every raw file the evaluation reads, which identifies the data behind the figures. The NOAA archive ends 2025-08-27, about 14 months before the demand data ends; the Mesonet fills the gap from 2025-08-20 so that no station is left with a gap. The weather builder uses NOAA wherever a station has it and the Mesonet after that.

**Licences.** EIA data is public domain and the API terms allow it to be displayed, with no claim that EIA is the source of modified content. NOAA data is public domain. The Mesonet states that its materials are public domain and free for any lawful purpose, with credit appreciated. The dashboard credits the EIA values to the U.S. Energy Information Administration (Hourly Electric Grid Monitor, Form EIA-930), labels every cleaned series, forecast, error figure and SHAP value as computed by the author from EIA data, credits NOAA and the Mesonet for their dates, and carries no EIA logo or wording that implies endorsement. The code carries an MIT licence.

## Data and cleaning

**Measure.** The target is demand for each hour of the target day. Demand is the BA's reported figure, which is close to net generation minus net interchange. For a BA that exports a large share of what it generates, reported demand follows its exports ([SRP](#why-error-rises-in-2025-and-2026)).

**Which BAs.** The API route has 83 respondents. The 14 regional aggregates (13 regions and the Lower 48 total) are left out so that no aggregate is counted beside its members. The table counts BAs down to the 20 used; the selection rule is applied to 2019 and 2020 only, before any model is fitted.

<!-- table:ba_funnel -->
| Stage | BAs |
| --- | ---: |
| Respondents on the API route | 83 |
| Regional aggregates | 14 |
| BAs | 69 |
| Generation-only BAs with no demand series | 12 |
| BAs with a demand series | 57 |
| Fail the rule (BHBA, DOPD, GVL, NSB, PSEI, SEC, SWPW, TAL) | 8 |
| Pass the rule | 49 |
| Smaller than the 20 largest passing BAs | 29 |
| Used | 20 |
<!-- /table -->

The rule keeps a BA when it has demand from 2019-01-01 and at most 2% of the 17,544 hours of 2019 and 2020 are missing, null, zero or negative, then takes the 20 largest by mean demand. The 20, with the shares measured on those two years:

<!-- table:selection -->
| Rank | Code | Name (EIA) | Mean demand, MW | Missing or null, % |
| ---: | --- | --- | ---: | ---: |
| 1 | PJM | PJM Interconnection, LLC | 89,580 | 0.27 |
| 2 | MISO | Midcontinent Independent System Operator, Inc. | 72,598 | 0.03 |
| 3 | ERCO | Electric Reliability Council of Texas, Inc. | 43,583 | 0.00 |
| 4 | SWPP | Southwest Power Pool | 30,299 | 0.00 |
| 5 | SOCO | Southern Company Services, Inc. - Trans | 26,563 | 0.14 |
| 6 | CISO | California Independent System Operator | 24,631 | 0.07 |
| 7 | TVA | Tennessee Valley Authority | 17,807 | 0.69 |
| 8 | NYIS | New York Independent System Operator | 17,444 | 0.00 |
| 9 | FPL | Florida Power & Light Co. | 14,735 | 0.11 |
| 10 | ISNE | ISO New England | 13,332 | 0.00 |
| 11 | DUK | Duke Energy Carolinas | 11,853 | 0.13 |
| 12 | CPLE | Duke Energy Progress East | 6,928 | 0.27 |
| 13 | BPAT | Bonneville Power Administration | 6,313 | 0.00 |
| 14 | FPC | Duke Energy Florida, Inc. | 6,244 | 0.13 |
| 15 | PSCO | Public Service Company of Colorado | 5,487 | 0.00 |
| 16 | PACE | PacifiCorp East | 5,468 | 0.00 |
| 17 | NEVP | Nevada Power Company | 4,271 | 0.10 |
| 18 | LGEE | LG&E and KU Services Company as agent for Louisville Gas and Electric Company and Kentucky Utilities Company | 3,916 | 0.64 |
| 19 | SRP | Salt River Project Agricultural Improvement and Power District | 3,561 | 0.96 |
| 20 | AZPS | Arizona Public Service Company | 3,453 | 0.14 |
<!-- /table -->

All 20 report from 2019-01-01 to the end of the data, so the panel is balanced. The 20 are not a random sample of BAs: they are the largest, so the results describe large, well-reporting BAs.

**Time.** EIA's series are on UTC and hour-ending. Hourly totals built that way match EIA's daily totals on all 28 days checked for ERCOT, PJM and CAISO; reading the stamps as hour-beginning does not match. Calendar inputs (hour, weekday, month, holidays) use each BA's local clock. EIA's time-zone table is not always the clock a BA's demand follows: MISO is set to America/Chicago and SOCO to America/New_York, and SRP and AZPS to America/Phoenix because Arizona does not observe daylight saving. `scripts/timezone_check.py` compares each BA's daily demand profile with its zone and confirms the daylight-saving shift for 18 of the 20; TVA and NEVP show none and are treated as noise.

**Weather.** Each BA is linked to one to four airport stations near its load centres (42 stations in all; Phoenix serves both SRP and AZPS; FPL uses Miami and Fort Lauderdale), weighted equally. An observation counts toward the hour it falls in, stamped to the hour that ends it. NOAA readings flagged suspect or erroneous (quality codes 2, 3, 6 and 7) are dropped. The Mesonet carries no quality flags, so readings that are physically impossible are set to missing; none were found. Every station has a temperature in at least 99.3% of hours and every BA in at least 99.46% ([weather_coverage.csv](../results/tables/weather_coverage.csv), [weather_coverage_by_ba.csv](../results/tables/weather_coverage_by_ba.csv)). Over the year where the two sources overlap, 344,418 temperature readings match; they differ by 0.03 °C on average, 119 differ by more than 0.5 °C and 9 by more than 2 °C, and none is filtered ([weather_source_overlap.csv](../results/tables/weather_source_overlap.csv)). In July and August 2023 ERCOT demand correlates 0.97 with the temperature of the previous hour, and PJM 0.92, with no stronger correlation at a later hour ([demand_temperature_correlation.csv](../results/tables/demand_temperature_correlation.csv)).

**Impossible readings.** The raw demand series contain values that cannot be demand: PJM at about 2.1 billion megawatthours (2^31), SWPP at 3.6 million, TVA at -9.9 million, and spikes or drop-outs of 5 to 10 times in SRP, NEVP, PACE, AZPS, PSCO and FPL. A reading is set to missing when it is not positive or falls outside 0.4 to 2.0 times the median of the previous 15 days. The rule looks backward only, so nothing after a reading decides whether it is kept, and raw files are untouched. It removes 332 of 1,374,422 hours across the 20 BAs, never more than 0.11% of a BA's hours (the largest share is AZPS at 0.106%). It removes nothing during Uri and three hours during Elliott: two FPC readings that look genuine (counted below) and one impossible SRP reading of 35,836 MW. Against the operator forecast, removed readings are usually hundreds of percent away from it, but 15 of the 227 hours removed on the 17 BAs that report one lie within 25% of it and look genuine (the June 2025 ISNE heat wave, SRP's May 2026 ramp, FPC during Elliott, and FPC on 2024-10-10 when demand dropped). After cleaning, 72 of the scored hours have a previous-day error above 100%, mostly isolated dips in SRP, AZPS and FPC that fall just inside the thresholds ([residual_outliers.csv](../results/tables/residual_outliers.csv)). The thresholds are round numbers chosen by eye on the full dataset, not fitted on training data. Three settings change no model's error by more than 0.03 points:

<!-- table:thresholds -->
| Thresholds | Hours | Yesterday | Regression | LightGBM |
| --- | ---: | ---: | ---: | ---: |
| 0.4/2.0 (used) | 1,015,245 | 7.412 | 8.767 | 5.183 |
| 0.3/3.0 | 1,015,613 | 7.422 | 8.774 | 5.198 |
| 0.5/1.8 | 1,014,062 | 7.397 | 8.755 | 5.178 |
<!-- /table -->

**Missing values.** Inputs are never interpolated, because filling a gap would use values after the origin. A gap stays missing. A target hour with no actual demand is not scored, and an hour where a forecast's input is missing is dropped for every forecast, so that all forecasts are scored on the same hours.

**From target hours to scored hours.** Each count adds up:

<!-- table:funnel -->
| Stage (variant A) | Hours |
| --- | ---: |
| Target hours in the test period (2021-01-01 to 2026-11-01) | 1,022,880 |
| Actual demand missing (a gap, or set to missing by the cleaning rule) | 1,532 |
| Actual demand present but a forecast input missing | 6,103 |
| Scored: actual demand and all five forecasts (20 BAs) | 1,015,245 |
| Of those, with a day-ahead forecast reported to EIA (17 BAs) | 856,669 |
| Of those, without one (FPC, PSCO, LGEE) | 158,576 |
<!-- /table -->

The three BAs without a usable operator forecast (FPC, PSCO, LGEE) are scored in every comparison except those with the operator forecast, which is why those comparisons cover 17 BAs. The operator forecast is excluded for them because it is far from the actual demand: in 2023 FPC's is 22.6% off and biased low by the same amount, PSCO's is 16.5% off (biased high by 10.0%) and LGEE's is 8.0% off (biased low by 7.6%), against 2.0% to 5.6% for the other 17 ([eia_forecast_2023.csv](../results/tables/eia_forecast_2023.csv)).

## Forecast design

**Target and origin.** At 10:00 local time on day *d*, each BA's hourly demand for every hour of day *d*+1 is forecast, so the horizon is 14 to 37 hours (14 to 36 on a 23-hour day and 14 to 38 on a 25-hour day). 10:00 is the time of day when day-ahead markets close their offers and bids (see [Forecast origin](#forecast-origin)); it is not a guarantee that every BA could have finished a forecast by then.

**Inputs.** Every input is at or before its cutoff on day *d*: demand through the stamp ending 08:00 and weather through the stamp ending 09:00.
- *Demand:* the latest reading; the mean and maximum of the previous 24 hours; the mean of the previous 168 hours; the change over 3 hours; the same hour 1, 2, 3 and 4 weeks earlier; and the same hour yesterday where it is known by 08:00, otherwise two days earlier.
- *Calendar:* hour, weekday, month, day of year (as sine and cosine), holiday, the day before and after a holiday, hours in the local day, horizon, and the age of the yesterday value.
- *Weather, variant A:* temperature, dew point and wind at 09:00, the 24-hour mean temperature and its 3-hour change.
- *Weather, variant B only:* temperature, dew point and wind at the target hour, and the target day's maximum, minimum and mean temperature and heating and cooling degrees.

**Leakage rules.** Each is tested on synthetic series where a value equals its own position on the hourly grid, so any input larger than its cutoff read the future ([tests/test_features.py](../tests/test_features.py)): no demand input is after the 08:00 stamp; no variant A weather input is after the 09:00 stamp; calendar inputs come from local time and UTC hour of day is never an input; a test that changes every value after the cutoffs leaves that date's inputs unchanged, and a companion test confirms that perturbation test can fail; training rows end two days before the first test origin of a block, because a row's targets run to local midnight of the day after next; scaling and gap handling use the training window only; and the operator forecast is a benchmark and never an input.

## Models

| Forecast | Inputs | Purpose |
|---|---|---|
| Same hour last week | Demand at the same hour one week earlier | A floor any model must beat |
| Same hour yesterday | Demand at the same hour one day earlier where known at the origin, otherwise two days earlier | A second floor; neither floor dominates for every BA |
| Average of the last four same weekdays | Mean of the same hour over the previous four same weekdays | A smoother floor |
| Regression | Hour-of-week effects, holiday flags, heating and cooling degrees by six-hour block, fitted separately for each BA | A simple structured benchmark. It is weak: worse than yesterday's demand in variant A |
| Ridge regression | The inputs LightGBM uses, with hour, weekday, month and BA effects, and heating and cooling degrees by six-hour block | A stronger simple benchmark. The penalty is chosen from 0.1, 1, 10 and 100 on the last 90 days of each training window, fitting only on rows whose targets end before that stretch starts |
| LightGBM | The inputs listed above, with the BA as a category | The model under test |
| LightGBM without weather | LightGBM with no weather inputs at all | Shows what weather adds. It does not depend on the variant |
| Operator forecast | The day-ahead forecast each BA reports to EIA | A benchmark for the 17 BAs that report a usable one |

LightGBM is one global model across BAs. Demand inputs and the target are divided by each BA's mean demand over the training window, then scaled back, so a fit never sees the test period. The hyperparameters are set by judgment, not tuned: learning rate 0.05 with 400 rounds, 63 leaves, at least 200 rows per leaf, a 0.8 share of features and of rows per round, and an L2 penalty of 1.0. They are common defaults for a table of about a million rows (`LGBM_PARAMS` in `electricity/models.py`) and are the same in every block. The seed is fixed. On the small table in `tests/test_models.py`, four retrains give identical forecasts.

## Validation

The first training period is 2019 and 2020. Testing runs from 2021-01-01 to 2026-11-01 in calendar quarters (24 blocks, the last covering October 2026 only), each forecast by models refitted on all earlier origin dates (an expanding window that starts in 2019). 2026 covers January to October. COVID (2020) sits in the first training period, not in the test set.

**The test period was used during iteration.** The forecasts are leakage-free walk-forward forecasts, but 2021 to 2026 is not a holdout that was scored once. Choices made after looking at it: the cleaning thresholds, the 10:00 forecast origin, keeping the training-mean scaling, how SRP is shown, and the two weather variants. A later first training period was not tried.

**Regimes.** Winter Storm Uri is 2021-02-13 to 02-19; Winter Storm Elliott 2022-12-21 to 12-27; the Texas heat is ERCOT only, 2023-06-20 to 09-15; holidays are U.S. federal holidays and the day before and after each. A regime is any hour that matches it, so regimes overlap: on the 17-BA sample 1,224 Uri hours also count as holiday-adjacent (Presidents' Day). "All other days" is everything in none of them. Demand reported during Uri is load actually served, so rotating outages are part of the error in those days.

**Intervals.** The comparisons between forecasts carry bootstrap intervals from resampling whole calendar weeks with all BAs in a week kept together, because weather moves many BAs at once and hours within a week are strongly related. The statistic is the paired difference in error over the hours both forecasts cover.

## Results

All 20 BAs, errors in percent, each forecast scored on the same hours. The "17 BAs" columns cover the BAs that report a usable operator forecast.

<!-- table:headline -->
| Forecast | A, 20 BAs | B, 20 BAs | A, 17 BAs | B, 17 BAs |
| --- | ---: | ---: | ---: | ---: |
| Same hour last week | 9.48 | 9.48 | 9.23 | 9.23 |
| Same hour yesterday | 7.41 | 7.41 | 7.18 | 7.18 |
| Average of the last four same weekdays | 9.17 | 9.17 | 9.00 | 9.00 |
| Regression (hour of week, holidays, temperature) | 8.77 | 7.69 | 8.66 | 7.65 |
| LightGBM | 5.18 | 3.32 | 4.99 | 3.21 |
| Operator forecast reported to EIA | n/a | n/a | 3.75 | 3.75 |
<!-- /table -->

A ridge regression on the same inputs, and LightGBM without weather, on the hours all seven forecasts cover:

<!-- table:extra_models -->
| Forecast | Error, A | Error, B | Peak-hour error, A | Peak-hour error, B |
| --- | ---: | ---: | ---: | ---: |
| Same hour yesterday | 7.41 | 7.41 | 8.02 | 8.02 |
| Regression (hour of week, holidays, temperature) | 8.76 | 7.69 | 9.83 | 7.78 |
| Ridge regression, same inputs as LightGBM | 5.98 | 4.75 | 6.68 | 5.03 |
| LightGBM without weather | 5.83 | 5.83 | 6.88 | 6.88 |
| LightGBM | 5.18 | 3.32 | 6.33 | 4.06 |

All rows: the 1,013,138 hours that all seven forecasts cover (variant A).
<!-- /table -->

Differences in error between forecasts, in percentage points, with a bootstrap interval (a negative first-minus-second value means the first is more accurate):

<!-- table:bootstrap_pooled -->
| Comparison (first minus second) | Hours | Variant A | Variant B |
| --- | --- | ---: | ---: |
| LightGBM minus yesterday's demand | 20 BAs | -2.23 [-2.32, -2.14] | -4.09 [-4.20, -3.97] |
| LightGBM minus regression | 20 BAs | -3.58 [-3.72, -3.45] | -4.37 [-4.51, -4.24] |
| LightGBM minus ridge | 20 BAs | -0.81 [-0.86, -0.75] | -1.44 [-1.49, -1.38] |
| Ridge minus yesterday's demand | 20 BAs | -1.42 [-1.51, -1.34] | -2.65 [-2.77, -2.54] |
| LightGBM minus LightGBM without weather | 20 BAs | -0.65 [-0.70, -0.61] | -2.51 [-2.60, -2.42] |
| LightGBM minus operator forecast | 17 BAs with an EIA forecast | +1.24 [+1.12, +1.36] | -0.54 [-0.64, -0.45] |
| Ridge minus operator forecast | 17 BAs with an EIA forecast | +2.03 [+1.91, +2.15] | +0.85 [+0.74, +0.95] |
| LightGBM minus operator forecast | 16 BAs with an EIA forecast, without SRP | +1.22 [+1.09, +1.34] | -0.56 [-0.67, -0.47] |
<!-- /table -->

LightGBM is the most accurate of the seven in both variants, in every year and in all 20 BAs (against ridge and against LightGBM without weather; [extra_models_by_ba.csv](../results/tables/extra_models_by_ba.csv)). A linear model on the same inputs gets about 64% of the way from yesterday's demand to LightGBM in variant A, so the margin of a flexible model over a good simple model is modest. Weather known at the forecast time is worth 0.65 points to LightGBM and perfect target-day weather 2.5 points.

Against the operator forecast, variant A, with the intervals by year:

<!-- table:bootstrap_eia_by_year -->
| Period | Variant A | Variant B |
| --- | ---: | ---: |
| 2021 | +1.78 [+1.58, +1.97] | -0.24 [-0.35, -0.15] |
| 2022 | +1.67 [+1.45, +1.90] | -0.32 [-0.43, -0.21] |
| 2023 | +1.50 [+1.34, +1.65] | -0.21 [-0.29, -0.12] |
| 2024 | +1.36 [+1.16, +1.57] | -0.28 [-0.38, -0.17] |
| 2025 | +0.76 [+0.53, +0.98] | -0.75 [-0.91, -0.60] |
| 2026 | +0.20 [-0.30, +0.64] | -1.62 [-2.16, -1.22] |
| 2021 to 2026 | +1.24 [+1.12, +1.36] | -0.54 [-0.64, -0.45] |
<!-- /table -->

With weather known at the forecast time, the operator forecast is more accurate over 2021 to 2026 and in every year up to 2025. In 2026 the gap is 0.20 points with an interval that includes zero, so the two cannot be told apart. In variant B, LightGBM is more accurate than the operator forecast, but that comparison is not like for like: LightGBM sees the target day's weather and the operator forecast does not.

By year, on the 17 BAs:

<!-- table:by_year -->
| Year | Same hour yesterday | LightGBM, A | LightGBM, B | Operator forecast |
| --- | ---: | ---: | ---: | ---: |
| 2021 | 7.20 | 4.89 | 2.87 | 3.12 |
| 2022 | 7.37 | 4.97 | 2.98 | 3.30 |
| 2023 | 6.80 | 4.69 | 2.98 | 3.19 |
| 2024 | 6.79 | 4.78 | 3.13 | 3.41 |
| 2025 | 7.16 | 4.98 | 3.48 | 4.23 |
| 2026 | 7.89 | 5.76 | 3.92 | 5.55 |
<!-- /table -->

By regime, on the 17 BAs:

<!-- table:regimes -->
| Regime | Hours | Same hour yesterday | LightGBM, A | LightGBM, B | Operator forecast |
| --- | ---: | ---: | ---: | ---: | ---: |
| Winter Storm Uri (2021-02-13 to 02-19) | 2,855 | 6.74 | 6.19 | 3.86 | 5.55 |
| Winter Storm Elliott (2022-12-21 to 12-27) | 2,709 | 10.65 | 7.97 | 4.81 | 4.40 |
| Texas heat, ERCOT (2023-06-20 to 09-15) | 2,112 | 4.15 | 5.31 | 3.19 | 2.04 |
| Holidays and adjacent days | 75,318 | 7.91 | 5.45 | 3.59 | 3.96 |
| Calm (none of the above) | 776,228 | 7.11 | 4.93 | 3.17 | 3.73 |
<!-- /table -->

In the Texas heat the previous day's demand is more accurate than LightGBM in variant A. Uri's error partly measures outages.

## Why error rises in 2025 and 2026

Every forecast's error rises in 2025 and 2026, the operator forecast most. The rise is in the tails, not the typical hour: median error moves little while the share of hours with an error above 20% grows.

<!-- table:error_rise -->
| Group | Year | LightGBM | Yesterday | Operator forecast | LightGBM median | Operator median | LightGBM, % of hours over 20 | Operator, % of hours over 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 BAs | 2024 | 4.78 | 6.79 | 3.41 | 3.49 | 2.44 | 1.3 | 0.6 |
| 17 BAs | 2025 | 4.98 | 7.16 | 4.23 | 3.56 | 2.67 | 1.9 | 1.6 |
| 17 BAs | 2026 | 5.76 | 7.89 | 5.55 | 3.99 | 2.68 | 3.2 | 4.3 |
| 17 BAs without SRP and AZPS | 2024 | 4.76 | 6.91 | 3.39 | 3.48 | 2.40 | 1.3 | 0.6 |
| 17 BAs without SRP and AZPS | 2025 | 4.86 | 7.16 | 3.85 | 3.44 | 2.47 | 1.8 | 1.3 |
| 17 BAs without SRP and AZPS | 2026 | 5.22 | 7.41 | 4.75 | 3.77 | 2.39 | 2.0 | 3.1 |
| SRP | 2024 | 4.85 | 5.87 | 3.15 | 3.64 | 2.56 | 1.1 | 0.0 |
| SRP | 2025 | 5.19 | 6.85 | 3.92 | 3.91 | 3.31 | 1.6 | 0.2 |
| SRP | 2026 | 11.96 | 13.81 | 13.71 | 7.80 | 6.01 | 18.5 | 22.1 |
<!-- /table -->

- **SRP.** Its error for every forecast is far above the rest in 2026. Its net generation stays steady while its net exports vary, and reported demand falls with them. From 2026-05-03 to 05-05 net generation stayed between 5,124 and 5,168 MW a day while net interchange rose from 1,919 to 3,272 MW and reported demand fell from 3,245 to 1,966 MW; on May 4 and 5 the operator forecast expected 3,996 and 3,649 MW and LightGBM 3,587 and 2,728 MW ([srp_may_2026_daily.csv](../results/tables/srp_may_2026_daily.csv), local-day means). The accounting behind the figures (pseudo-ties, dynamic scheduling, market participation) is not known, so the cause is narrowed and not proven. SRP generates about 70% more than the demand it reports, and the daily changes in its demand follow the daily changes in interchange far more closely in 2026:

<!-- table:srp_flows -->
| BA | Year | Interchange, % of net generation | Demand daily change, sd % of mean | Net generation daily change, sd % of mean | Correlation of demand and interchange changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| SRP | 2021 | 43.9 | 6.0 | 11.8 | -0.13 |
| SRP | 2022 | 41.4 | 9.7 | 9.1 | -0.14 |
| SRP | 2023 | 42.7 | 14.5 | 8.6 | -0.18 |
| SRP | 2024 | 40.5 | 5.0 | 4.3 | -0.19 |
| SRP | 2025 | 40.4 | 6.4 | 4.7 | -0.24 |
| SRP | 2026 | 42.2 | 10.7 | 5.4 | -0.60 |
| AZPS | 2021 | -58.9 | 5.5 | 8.1 | -0.56 |
| AZPS | 2022 | -50.3 | 5.5 | 7.9 | -0.52 |
| AZPS | 2023 | -67.3 | 4.9 | 8.5 | -0.47 |
| AZPS | 2024 | -61.1 | 5.0 | 7.6 | -0.54 |
| AZPS | 2025 | -61.8 | 6.8 | 9.5 | -0.54 |
| AZPS | 2026 | -58.7 | 9.1 | 12.8 | -0.38 |
<!-- /table -->

  Days are the mean of the hours where demand, net generation and interchange are all present and positive. SRP stays in the headline results because the BA list is fixed by a rule applied to 2019 and 2020. Without SRP the comparisons keep their rankings and no pooled error moves by more than 0.2 points ([overall.csv](../results/tables/overall.csv), the `without_SRP` samples).
- **The operator forecast** gets worse from 2024 to 2026 by more than the previous-day forecast does in SRP and in six other BAs: SWPP, CPLE, AZPS, CISO, PACE and NEVP ([error_rise_by_ba.csv](../results/tables/error_rise_by_ba.csv)).
- **Not model decay.** Without SRP and AZPS, LightGBM and the previous-day forecast rise together, so the ratio of their errors is stable ([error_rise_ratio.csv](../results/tables/error_rise_ratio.csv)).
- **Growth.** LightGBM runs a little low where demand has grown beyond the training level (ERCO up 24% from 2021 to 2025, AZPS 26%, SRP 15%, FPL 14%). Across 480 BA-quarters, the further the previous year's demand sits above the training mean, the lower the bias, with a correlation of -0.44 ([growth_bias_bins.csv](../results/tables/growth_bias_bins.csv)). This is a correlation and not a proven cause.
- **The newest data.** The last four days of data (2026-10-30 to 11-02) are not worse than the preceding four weeks for the median BA (the median ratio of the two errors is 0.72); four BAs run higher, SRP most (1.59 times, after a poor October throughout), then FPL, BPAT and PACE ([newest_days.csv](../results/tables/newest_days.csv)).

## Forecast origin

10:00 local is the time of day when most day-ahead markets close their offers and bids. Each market's close, from its operator's own documents:

| Market (BA) | Offers and bids close | Results | Source |
|---|---|---|---|
| NYISO (NYIS) | 5:00 a.m. ET for load forecasts and bids (4:50 a.m. for some external transaction bids, 4:45 a.m. for bids with revised fuel information) | Schedule posted by 11:00 a.m. | Manual 11, Day-Ahead Scheduling Manual, version 10.1, effective 2026-09-03; Market Services Tariff, published 2026-10-29, sections 4.2.1.1 and 4.2.5 |
| ISO-NE (ISNE) | 10:30 a.m. ET | After the market clears | Manual M-11, revision 62, effective 2025-02-28, section 2.2.1 |
| MISO (MISO) | 10:30 a.m. Eastern Prevailing Time | Not checked | BPM-002 (Energy and Operating Reserve Markets), revision 26, effective 2025-09-30; Tariff section 39.1.1 |
| PJM (PJM) | 11:00 a.m. ET | By 1:30 p.m. | Manual 11, revision 137, effective 2026-07-28, sections 2.1 and 2.3.3A.1 |
| ERCOT (ERCO) | 10:00 a.m. CT: day-ahead market energy bids and offers "by 1000" | No later than 1:30 p.m. | Nodal Protocols, Section 4 (Day-Ahead Operations), dated 2026-07-10 |
| CAISO (CISO) | 10:00 a.m. PT | 1:00 p.m. | Business Practice Manual for Market Operations, version 101.1, 2025-03-26 |

Five of the six markets close between 10:00 and 11:00 in their own clock, so 10:00 is a representative time of day. It does not show that a forecast with a 10:00 cutoff was feasible everywhere: it is exactly at the close for ERCOT and CAISO, 30 minutes before ISO-NE's close, later than NYISO's, and for MISO (set to Central here) 10:00 CT is 11:00 EPT, after the 10:30 EPT close. Only 6 of the 20 BAs sit in these markets; the other 14 are utilities for which no such deadline was found (SWPP was not checked). EIA states that it publishes the operator forecast by 11:00 ET but not when each BA made it, so its information cannot be matched to the model's. Operators can extend a deadline after an outage, so these are scheduled times.

Moving the origin from 10:00 to 06:00 or 12:00 changes LightGBM's error by 0.43 and 0.24 points and changes no ranking (2024 onward, variant A):

<!-- table:origin -->
| Origin | LightGBM, 20 BAs | Yesterday, 20 BAs | LightGBM, 17 BAs | Yesterday, 17 BAs | Operator forecast, 17 BAs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 06:00 | 5.77 | 7.96 | 5.57 | 7.71 | 4.33 |
| 10:00 | 5.34 | 7.49 | 5.14 | 7.25 | 4.33 |
| 12:00 | 5.10 | 7.26 | 4.90 | 7.01 | 4.33 |
<!-- /table -->

A later origin helps because the model sees more of the day's demand, mostly in the morning hours of the target day. A 12:00 origin would be after EIA's stated publication time for every BA and after every close above.

## What the model uses

One LightGBM model is fitted on data through 2024 and explained on 60,000 randomly drawn hours of 2025, with LightGBM's built-in tree SHAP. Values are the mean absolute contribution to a forecast, as a share of each BA's average demand. Shares of the total:

<!-- table:shap_groups -->
| Input group | Variant A | Variant B |
| --- | ---: | ---: |
| Recent demand and lags | 60% | 45% |
| Calendar and holidays | 20% | 17% |
| Weather known at the forecast time | 18% | 5% |
| Which balancing authority | 3% | 6% |
| Target-day weather (variant B only) | n/a | 28% |
<!-- /table -->

The hour yesterday is the largest single input in both variants, so the model works as a correction to yesterday's demand using weather and calendar. Related inputs share credit, so the groups are more reliable than single inputs. This is one model and one year, and the group ranking was not checked across years. The BA identity matters little, so one model across BAs is not memorising each BA.

## Scaling by a recent level

LightGBM divides demand by each BA's training-window mean. Dividing by the mean of the previous 168 hours instead removes the link between bias and recent growth (the correlation of bias with the previous year's level falls from -0.44 to 0.04 across the 480 BA-quarters) but moves the overall bias from -0.37 to +0.47 in variant A and from -1.00 to +0.26 in variant B. Overall error is unchanged in both variants, the scaling helps about half the BAs, and its effect in Uri and Elliott changes sign between variants. The headline keeps the training-mean scaling.

<!-- table:scaling_summary -->
| Measure | A, training mean | A, previous week | B, training mean | B, previous week |
| --- | ---: | ---: | ---: | ---: |
| Error | 5.19 | 5.19 | 3.32 | 3.34 |
| Bias | -0.37 | 0.47 | -1.00 | 0.26 |
| Peak-hour error | 6.33 | 6.04 | 4.07 | 3.74 |
| Error, Winter Storm Uri | 6.38 | 5.95 | 3.84 | 4.80 |
| Error, Winter Storm Elliott | 8.34 | 7.96 | 5.05 | 5.56 |
| Error, Texas heat, ERCOT | 5.31 | 3.53 | 3.19 | 2.37 |
| Error, Holidays and adjacent days | 5.63 | 5.69 | 3.72 | 3.83 |
| Error, Calm | 5.13 | 5.14 | 3.28 | 3.28 |
<!-- /table -->

## Limitations

- **History.** The API starts in 2019, so extreme events are few (Uri and Elliott are the main ones).
- **Revisions.** Values are the latest revised ones, so a backtest is easier than a live system's.
- **Weather.** The headline uses observed weather up to 09:00, not weather forecasts. Variant B uses observed weather for the target day and is an upper bound.
- **Operator forecast.** Its information time is unknown and differs between BAs. It is unusable for three BAs.
- **Reporting quirks.** Pseudo-ties and dynamic scheduling can distort a BA's demand, and SRP's reported demand in 2026 follows its exports. Distribution-connected generation such as rooftop solar is generally excluded from demand.
- **Test period.** It informed several choices (see [Validation](#validation)).
- **Cleaning.** The thresholds were chosen by eye, 15 genuine-looking hours are removed, and a few dozen isolated dips remain.
- **Growth.** LightGBM under-forecasts BAs whose demand has grown past the training level, by up to about 1%.
- **Scope.** The results describe large, well-reporting BAs and the model against baselines and published forecasts; they do not describe outperforming grid operators.

## Reproducibility

**Run order.**

```
python scripts/build_respondents.py    # classify the API's respondents
python scripts/completeness.py         # the BA selection rule on 2019 to 2020
python scripts/ingest.py --types D,DF  # raw demand and day-ahead forecasts (needs EIA_API_KEY)
python scripts/ingest_balance.py --bas SRP,AZPS --types NG,TI
python scripts/weather_stations.py && python scripts/ingest_weather.py && python scripts/ingest_iem.py && python scripts/weather_hourly.py
python scripts/run_evaluation.py --variant A    # and --variant B
python scripts/run_extra_models.py --variant A --models ridge       # and B; and --models lightgbm_noweather
python scripts/run_shap.py --variant A          # and B
python scripts/build_results.py && python scripts/bootstrap_intervals.py
python scripts/render_doc_tables.py             # rewrite the tables in the docs
```

**Determinism.** On the small table in `tests/test_models.py`, the same inputs and seed give identical forecasts and a different seed gives different forecasts. LightGBM fits on all threads (`num_threads: 0`), so two full evaluations can differ in the last digits of some LightGBM forecasts (a rounding difference, far below any figure reported here), and the overall, by-BA and by-regime tables are the same in both runs. The evaluation takes about 17 minutes for variant A and 27 for variant B on a desktop machine.

**Data version.** [data_manifest.csv](../results/tables/data_manifest.csv) lists the SHA-256 of every raw input.

**Software.** Python 3.14 is the development version; CI runs the tests on 3.12, 3.13 and 3.14. Packages are pinned in `requirements.txt`. The model results come from LightGBM 4.7.0 and pandas 3.1.0. With the pinned pandas 3.1.1 the results tables rebuild unchanged, and a fresh variant A evaluation gives the same overall, by-BA and by-regime tables.

**Tests.** `python -m pytest` runs offline in under a minute. They cover the leakage rules, the cleaning rule, the scoring and walk-forward code, the models (including determinism), the fetchers against mocked replies, and the committed results: the daily error sums recompute the headline tables, the dashboard data equals the tables, and the figures in these documents equal the files. They do not cover page layout or the live deployment.

## Deployment notes

The dashboard is a static site: `site/` holds the page, a small stand-in for the charting runtime, and the 3.9 MB of results it reads. [render.yaml](../render.yaml) declares it as a Render static site with `site/` as the publish directory, so there is nothing to build or run and no hosted memory figure to report. After a figure changes, rebuild with `scripts/build_dashboard_data.py` and `scripts/build_site.py`, commit `site/`, and push. The page loads the d3 charting library from a CDN, so a viewer needs a network connection. The dashboard reads only the committed files and never calls a live service.
