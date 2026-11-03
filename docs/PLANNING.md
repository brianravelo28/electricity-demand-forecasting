# Electricity Demand Forecasting: Planning Doc

Nov 3, 2026 · @Brian Blitz

## Goal and framing

The project tests whether a gradient-boosted model forecasts next-day hourly electricity demand better than naive, seasonal-naive and simple regression baselines, across US balancing authorities (BAs).

- **Claims:** out-of-sample accuracy versus the baselines, reported overall, by balancing authority and by regime (extreme cold, heat waves, holidays, calm).
- **Does not claim:** a forecast that beats the grid operators' own forecasts in production, or anything about prices or trading.
- **Why demand:** it is driven by weather, hour of day, day of week and holidays, so it is forecastable. The question is how much a flexible model adds over simple structure.
- **Why this project:** EIA data is public domain and its API terms allow displaying it, so the dashboard can be public. See "Publishing rules".

## Target definition

The target is hourly demand (megawatthours, EIA series type `D`) for each of the 24 hours of the next local day, forecast at 10:00 local time on day t.

- **Forecast origin:** 10:00 in each BA's primary local time zone. Targets are 14 to 37 hours ahead. This mirrors a day-ahead schedule and is close to EIA's own day-ahead forecast, which it publishes by 11:00 ET. Confirm the origin before building features.
- **Clock:** the API's `hourly` series is on UTC (checked: ERCOT peaks at 23:00 on 2023-07-18, which is 6 pm Central, and the daylight-saving weekend returns exactly 72 hours). Calendar features (hour, weekday, holidays) use each BA's local time, not UTC.
- **Period stamp:** each stamp marks the hour that ends at that time (hour-ending). Hourly totals built that way match EIA's daily totals on all 28 days checked for ERCOT, PJM and CAISO. Reading the stamps as hour-beginning does not match.
- **Error metrics:** MAPE and RMSE in MW per BA, RMSE divided by mean demand to compare across BAs, MAE at each day's peak hour, and bias.

## Universe and data

The universe is the largest BAs by average demand with complete series, selected by a rule written down before modeling.

| Source | Role | Known limits |
| --- | --- | --- |
| EIA API v2, `electricity/rto/region-data` | Hourly demand (`D`), day-ahead forecast (`DF`), net generation, interchange | Free key; 5,000 rows per page; rate limits unpublished (key can be suspended); history starts 2019-01-01 |
| NOAA NCEI (hourly surface observations) | Temperature, humidity, wind per BA | Public domain under the NCEI open data policy; access route and token still to confirm |
| US federal holiday calendar | Holiday flags | Built from a published rule, no source data needed |

- **Respondents:** the route has 83: 69 BAs and 14 regional aggregates (13 regions plus the Lower 48 total). Twelve BAs are generation-only and have no demand series, leaving 57 BAs with demand. An aggregate is never included alongside its member BAs.
- **Time zones:** among the 57, 20 are Eastern, 17 Pacific, 8 Central, 7 Arizona and 5 Mountain. Arizona does not observe daylight saving, so it maps to `America/Phoenix`, not Mountain. The source is EIA's reference table. Spot-check it, since some assignments look odd (WACM, the Rocky Mountain region, is listed under Arizona).
- **Selection rule (draft):** using 2019 to 2020 data only, keep BAs with `D` present for the whole period and at most 2% of hours missing or null, and rank by mean demand. Start with about 20, then widen. 51 of the 57 have demand from 2019 and were still active in April 2026. BAs that retire after 2020 (so far AEC, NSB, WACM and WAUW) stay in until they stop reporting, so the panel is unbalanced. BAs that start after 2020 are out. Gaps in later years are flagged, not used for selection.
- **Gaps:** in 2023, ERCOT was complete, CAISO had 2 null values, and PJM had 1 missing hour and 24 nulls. Keep raw values untouched in storage and add flags. Decide gap handling (interpolate short gaps, drop long ones) from training data only.
- **Ingestion is separate from analysis:** one script downloads raw data to parquet and everything else reads only from those files. The public dashboard also reads saved files and never calls the API.

## Models and features

Four models are compared, so that any gain from machine learning is measured against structure the load-forecasting literature already uses.

| Model | Inputs | Purpose |
| --- | --- | --- |
| Seasonal-naive | Demand at the same hour one week earlier | Floor that any model must beat |
| Same-day-type average | Mean of the same hour over the previous 4 same weekdays | Smoother naive |
| Regression | Hour-of-week effects, holiday flags, temperature terms (heating and cooling degrees) | The simple structured benchmark, analogous to a HAR model |
| LightGBM | Demand lags known at the origin, rolling means, calendar, holidays, weather | Tests whether extra flexibility adds accuracy; SHAP shows what drives it |

- **EIA day-ahead forecast (`DF`):** a benchmark, not an input, because it is published by 11:00 ET, after the 10:00 origin. For 8 BAs EIA says its forecast is not meaningfully comparable to actual demand, so exclude those from this comparison.
- **Weather, two variants:** (A) weather observed up to the origin only; (B) observed weather for the target day, labeled as a perfect-forecast upper bound, since a live system would use weather forecasts. Report both, never mixed.

## Validation and leakage checklist

Validation is walk-forward with an expanding training window. The first training period is 2019 to 2020, and testing runs from 2021 with refits each quarter. Results are reported separately by regime.

- **Regimes in the test period:** Winter Storm Uri (February 2021), Winter Storm Elliott (December 2022), the summer 2023 heat in Texas, holidays, and calm stretches, each reported alone. COVID (2020) sits in the first training period, not the test set.
- **Uri caveat:** demand reported during Uri is load actually served. Rotating outages cut it, so errors in those days partly measure outages, not model quality.
- **Robustness (optional):** rerun with a later first training period and check whether the model ranking changes.

Leakage checklist:

1. Every feature uses data available at the origin (10:00 local on day t). Demand is published about an hour after each hour, so use demand only through the stamp at 08:00 local (the hour ending 08:00) to stay conservative.
2. Stamps are hour-ending, so a stamp at time T covers T minus 1 hour to T. Lags and the target day's 24 hours are built on that basis.
3. Calendar features come from local time. UTC hour-of-day is never a feature.
4. Weather in variant A stops at the origin. Variant B is labeled and kept separate.
5. EIA's day-ahead forecast is a benchmark only.
6. Training data ends at least 2 days before the first test origin, because targets reach 37 hours ahead.
7. Scaling, encoding, gap handling and outlier thresholds are fit on training data only.
8. The BA list is fixed by the rule above using 2019 to 2020 data only.
9. The API serves the latest revised values, not what was known in real time. This cannot be fixed, so it goes in the README.

## Publishing rules

Under the EIA API terms, the dashboard may display the data but may not claim EIA as the source of modified content.

- **Unaltered EIA values** are credited "Source: U.S. Energy Information Administration".
- **Everything computed here** (cleaned series, forecasts, error metrics, SHAP values) is labeled "computed by the author from EIA data".
- **No EIA logo** and no wording that implies EIA endorsement.
- **Third-party material** on EIA's site (such as photos) is not used.
- **NOAA weather** is credited to NOAA/NCEI.

## Known limitations for the README

These limits go in the README as stated facts, because naming them is part of what makes the project credible.

- **History:** the API starts in 2019, so the sample has few extreme events (Uri and Elliott are the main ones).
- **Revisions:** values are the latest revised ones, so backtests are easier than a live system's.
- **Imputation:** EIA fills some anomalous demand values itself, and the README states which cleaning rules were added.
- **Reporting quirks:** pseudo-ties and dynamic scheduling can distort a BA's demand. Distribution-connected generation such as rooftop solar is generally excluded.
- **Weather:** the main results use observed weather, not forecasts.
- **Scope:** conclusions are about the model versus baselines, not about outperforming operators.

## Work-checkpoint tasks

- [x] Confirm whether a period stamp marks the start or end of its hour (end)
- [x] Pull respondent metadata: which are BAs, aggregates or generation-only, and each BA's time zone
- [ ] Map EIA's five time-zone names to IANA zones and spot-check odd assignments
- [ ] Write the BA selection rule and run the completeness table
- [ ] Confirm the NOAA/NCEI hourly access route and token, and map stations to BAs
- [ ] Check when EIA's day-ahead forecast is published, per BA
- [ ] Write pseudocode for target and feature construction
- [ ] Settle the forecast origin (10:00 local is the draft)
