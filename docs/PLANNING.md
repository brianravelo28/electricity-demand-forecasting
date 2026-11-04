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

- **Forecast origin:** 10:00 in each BA's primary local time zone. Targets are 14 to 37 hours ahead. This is the time of day when most day-ahead markets close their offers and bids (10:00 to 11:00 in the market's own clock for ERCOT, CAISO, ISO-NE, MISO and PJM; NYISO closes at 5:00 ET), but it is not a guarantee that every BA could have finished a forecast in time (see the timing evidence under Results). Settled at 10:00 after a sensitivity check: moving it two to four hours changes LightGBM's error by about 0.2 to 0.4 percentage points and changes no ranking.
- **Clock:** the API's `hourly` series is on UTC (checked: ERCOT peaks at 23:00 on 2023-07-18, which is 6 pm Central, and the daylight-saving weekend returns exactly 72 hours). Calendar features (hour, weekday, holidays) use each BA's local time, not UTC.
- **Period stamp:** each stamp marks the hour that ends at that time (hour-ending). Hourly totals built that way match EIA's daily totals on all 28 days checked for ERCOT, PJM and CAISO. Reading the stamps as hour-beginning does not match.
- **Error metrics:** MAPE and RMSE in MW per BA, RMSE divided by mean demand to compare across BAs, MAE at each day's peak hour, and bias.

## Universe and data

The universe is the largest BAs by average demand with complete series, selected by a rule written down before modeling.

| Source | Role | Known limits |
| --- | --- | --- |
| EIA API v2, `electricity/rto/region-data` | Hourly demand (`D`), day-ahead forecast (`DF`), net generation, interchange | Free key; 5,000 rows per page; rate limits unpublished (key can be suspended); history starts 2019-01-01 |
| NOAA Integrated Surface Database (public S3 bucket `noaa-global-hourly-pds`) | Temperature, dew point and wind speed from airport stations | Public domain under the NCEI open data policy; no token. The archive has no observations after 2025-08-27 (checked in the bucket and in NCEI's data service on Nov 3, 2026) |
| Iowa Environmental Mesonet airport observations | Same station observations after 2025-08-27 | Site says its materials are public domain and free for any lawful purpose, credit appreciated. Over 11.5 months at 41 of the 42 stations (it has no Palm Beach), it matches ISD within 0.03 °C on average for temperature and dew point and within 0.03 m/s for wind |
| US federal holiday calendar | Holiday flags | Built from a published rule, no source data needed |

- **Respondents:** the route has 83: 69 BAs and 14 regional aggregates (13 regions plus the Lower 48 total). Twelve BAs are generation-only and have no demand series, leaving 57 BAs with demand. An aggregate is never included alongside its member BAs.
- **Time zones:** among the 57, 20 are Eastern, 17 Pacific, 8 Central, 7 Arizona and 5 Mountain. Arizona does not observe daylight saving, so it maps to `America/Phoenix`, not Mountain. The source is EIA's reference table. Spot-check it, since some assignments look odd (WACM, the Rocky Mountain region, is listed under Arizona). EIA also notes that MISO and LGEE report on Eastern Standard Time all year, so the reporting zone is not always the clock that demand follows. Calendar features need each BA's civil time zone, set by hand for BAs such as MISO that span two zones.
- **Selection rule (draft):** using 2019 to 2020 data only, keep BAs with `D` present for the whole period and at most 2% of hours missing or null, and rank by mean demand. Start with about 20, then widen. 51 of the 57 have demand from 2019 and were still active in April 2026. BAs that retire after 2020 (so far AEC, NSB, WACM and WAUW) stay in until they stop reporting, so the panel is unbalanced. BAs that start after 2020 are out. Gaps in later years are flagged, not used for selection.
- **Selection result** (`scripts/completeness.py`, 17,544 hours in 2019 to 2020): 49 of the 57 BAs pass. The 8 that fail are PSEI (53% of hours missing), SEC (39% of values zero or negative), TAL, GVL and DOPD (4% to 8% missing or null), and NSB, BHBA and SWPW (no data from 2019-01-01). The starting universe is the 20 largest passing BAs by mean demand: PJM, MISO, ERCO, SWPP, SOCO, CISO, TVA, NYIS, FPL, ISNE, DUK, CPLE, BPAT, FPC, PSCO, PACE, NEVP, LGEE, SRP and AZPS (down to about 3,450 MW). Each has under 1% of hours missing or null.
- **Weather stations (draft):** each of the 20 BAs maps to one to four airports in its load centers (42 stations in `scripts/weather_stations.py`; Phoenix serves SRP and AZPS), equal weighted. FPL uses Miami and Fort Lauderdale, because Palm Beach is not in the Mesonet archive. From 2019-01-01 to 2026-11-03, each station has a good temperature in at least 99.3% of hours, and every BA has one in at least 99.4% of hours from at least one of its stations. NOAA observations flagged suspect or erroneous are dropped. The airport choices are a judgement call to review.
- **Weather gap:** the ISD archive stops at 2025-08-27, about 14 months before the demand data ends. `scripts/ingest_iem.py` takes the same stations from the Iowa Mesonet from 2025-08-20 on (`data/raw/weather/obs_iem/`), early enough to cover the station whose ISD data ends first (2025-08-25). Mesonet has no quality flags, so readings that are physically impossible are set to missing, and none were found. Of 344,418 matched temperatures in the overlap year, 119 differ from ISD by more than 0.5 °C and 9 by more than 2 °C, and these are not filtered. After the switch, every BA has a good temperature in at least 99.6% of hours from at least one station. The hourly builder (`scripts/weather_hourly.py`) uses ISD wherever a station has it and Mesonet after that.
- **Gaps:** in 2023, ERCOT was complete, CAISO had 2 null values, and PJM had 1 missing hour and 24 nulls. Keep raw values untouched in storage and add flags. Features are never interpolated, because filling a gap uses values after the origin. Gaps stay missing, and target rows with a missing value are dropped.
- **Cleaning impossible readings (`electricity/cleaning.py`):** the raw demand has impossible values, for example PJM at about 2.1 billion MWh (2^31), SWPP at 3.6 million, TVA at -9.9 million, and spikes or drop-outs of 5 to 10 times in SRP, NEVP, PACE, AZPS, PSCO and FPL. A first full evaluation that only dropped non-positive readings was discarded, because these values dominated the errors (PJM showed 115% to 290%). The rule sets a reading to missing when it is not positive or falls outside 0.4 to 2.0 times the median of the previous 15 days. It looks backward only, so it uses nothing after the reading, and raw files are untouched. It removes under 0.11% of hours in every BA and nothing during Uri or Elliott. Checked against EIA's day-ahead forecast on the 17 BAs that have a usable one: removed readings are usually hundreds of percent away from the forecast, but 15 of 227 removed hours are within 25% of it and look genuine (the June 2025 ISNE heat wave, SRP's May 2026 ramp, FPC during Elliott, FPC on 2024-10-10, when demand dropped). The thresholds are fixed round numbers chosen by eye from the full dataset, not fit on training data. Rerunning variant A with 0.5/1.8 and 0.3/3.0 changed no model's overall error by more than 0.05 percentage points and did not change the ranking, so 0.4/2.0 was kept. About 72 of roughly a million scored rows still have a previous-day error above 100%, mostly isolated dips in SRP, AZPS and FPC that fall just inside the thresholds.
- **Ingestion is separate from analysis:** one script downloads raw data to parquet and everything else reads only from those files. The public dashboard also reads saved files and never calls the API.

## Models and features

Five models are compared, so that any gain from machine learning is measured against structure the load-forecasting literature already uses.

| Model | Inputs | Purpose |
| --- | --- | --- |
| Seasonal-naive | Demand at the same hour one week earlier | Floor that any model must beat |
| Previous-day naive | Demand at the same hour one day earlier when known by the origin, otherwise two days earlier | A second floor. In 2023 it beats seasonal-naive for ERCOT (6.1% against 8.1%), PJM (6.6% against 7.3%) and NYISO (6.6% against 7.5%) but not CAISO (7.0% against 6.6%), so neither dominates |
| Same-day-type average | Mean of the same hour over the previous 4 same weekdays | Smoother naive |
| Regression | Hour-of-week effects, holiday flags, temperature terms (heating and cooling degrees) | The simple structured benchmark, analogous to a HAR model |
| LightGBM | Demand lags known at the origin, rolling means, calendar, holidays, weather | Tests whether extra flexibility adds accuracy; SHAP shows what drives it |

- **EIA day-ahead forecast (`DF`):** a benchmark, not an input, because it is published by 11:00 ET, after the 10:00 origin. For 8 BAs EIA says its forecast is not meaningfully comparable to actual demand, so exclude those from this comparison. Checked on 2023 for the 20 selected BAs: the forecast is rarely missing (SRP 4.2%, FPL 0.6%) and almost never equals actual demand (under 0.4% of hours), so it was not overwritten with actuals. Its mean absolute percentage error is 2.0% to 5.6% for 16 BAs, against 5.8% to 13.5% for the seasonal-naive baseline. FPC (22.6%, biased low by 22.6%), PSCO (16.5%, biased high by 10.0%) and LGEE (8.0%, biased low by 7.6%) are far off their peers and are excluded from the benchmark comparison.
- **Weather, two variants:** (A) weather observed up to the origin only; (B) observed weather for the target day, labeled as a perfect-forecast upper bound, since a live system would use weather forecasts. Report both, never mixed.

## Target and feature construction (pseudocode)

All series are indexed by the UTC hour-ending stamp T, which covers the hour from T minus 1 hour to T, as in the EIA data. One row is one BA, one origin date d and one target hour. Local times use the BA's civil zone from `scripts/ba_timezones.py`.

```
inputs:  demand[ba][T], weather[ba][T] (temp, dew, wind), tz[ba], US federal holidays by local date

for each ba, for each local date d (the origin date):
    O   = utc(d at 10:00 local)             # forecast origin
    Ld  = utc(d at 08:00 local)             # last demand stamp allowed
    Wd  = utc(d at 09:00 local)             # last weather stamp allowed (variant A)
    M0  = utc(d+1 at 00:00 local)
    M1  = utc(d+2 at 00:00 local)
    targets = every hourly stamp T with M0 < T <= M1      # 24 stamps, 23 or 25 on a clock-change day

    # shared by all target hours of this (ba, d); every input is at or before Ld or Wd
    demand_last = demand[Ld]
    demand_mean_24h, demand_max_24h = mean, max of demand[Ld-23h .. Ld]
    demand_mean_168h = mean of demand[Ld-167h .. Ld]
    demand_change_3h = demand[Ld] - demand[Ld-3h]
    variant A weather: temp_last = temp[Wd];  temp_mean_24h over [Wd-23h .. Wd];
                       temp_change_3h;  dew_last;  wind_last

    for each T in targets:
        hour_local = local(T - 1h).hour                    # 0 to 23, the hour that T closes
        horizon    = (T - 1h) - O                          # about 14 to 37 hours
        lag_week_w = demand[T - 168h*w]  for w = 1..4      # always <= Ld
        lag_day    = demand[T - 24h]  if T - 24h <= Ld  else demand[T - 48h]
        lag_day_age = 1 or 2 days accordingly              # T - 24h is known only for hours ending by 08:00
        calendar   = hour_local, weekday, month, day-of-year sin and cos, holiday flag,
                     day-before and day-after holiday flags, hours in the local day (23, 24 or 25)
        variant B weather (labeled upper bound): temp[T], dew[T], wind[T], and the target day's
                     max, min and mean temperature, heating and cooling degrees
        y = demand[T]                                      # the target

baselines:
    seasonal_naive      = demand[T - 168h]
    previous_day_naive  = lag_day
    same_day_type_avg   = mean of demand[T - 168h*w], w = 1..4
    regression          = hour-of-week effects + holiday flags + heating and cooling degrees
                          (24-hour mean temperature in variant A, the target day's in variant B)
    eia_day_ahead (DF)  = benchmark only, for BAs not excluded above

modeling rules:
    * Missing, non-positive or implausible demand (see the cleaning rule under Universe and data)
      is set to missing. Features are never interpolated, because
      filling a gap would use values after the origin. A row is dropped when y or a baseline's
      input is missing, and every model is scored on the same remaining rows.
    * One global LightGBM across BAs with the BA as a category. Demand features and y are divided
      by the BA's mean demand over the current training window, then scaled back for scoring.
    * Training rows have origin date <= t0 - 2 days, where t0 is the first test origin date,
      because a row's targets run to local midnight of d+2.
```

The code is `electricity/features.py` and its checks are in `tests/test_features.py` (run `python -m pytest`). They build the features on a synthetic series whose value is its own timestamp and assert that no demand input is later than Ld, no variant A weather input is later than Wd, no training target is later than the first test origin, and the 23- and 25-hour days produce 23 and 25 targets. A further test changes every value after Ld and Wd and confirms that the features for that origin date do not move.

## Validation and leakage checklist

Validation is walk-forward with an expanding training window. The first training period is 2019 to 2020, and testing runs from 2021 with refits each quarter. Results are reported separately by regime.

- **Regimes in the test period:** Winter Storm Uri (February 2021), Winter Storm Elliott (December 2022), the summer 2023 heat in Texas, holidays, and calm stretches, each reported alone. COVID (2020) sits in the first training period, not the test set.
- **Uri caveat:** demand reported during Uri is load actually served. Rotating outages cut it, so errors in those days partly measure outages, not model quality.
- **Robustness (optional):** rerun with a later first training period and check whether the model ranking changes.

Leakage checklist:

1. Every feature uses data available at the origin (10:00 local on day t). Demand is published about an hour after each hour, so use demand only through the stamp at 08:00 local (the hour ending 08:00) to stay conservative.
2. Stamps are hour-ending, so a stamp at time T covers T minus 1 hour to T. Lags and the target day's 24 hours are built on that basis.
3. Calendar features come from local time. UTC hour-of-day is never a feature.
4. Weather in variant A stops at the 09:00 local stamp, an hour before the origin. Variant B is labeled and kept separate.
5. EIA's day-ahead forecast is a benchmark only.
6. Training data ends at least 2 days before the first test origin, because targets reach 37 hours ahead.
7. Scaling, encoding and gap handling are fit on training data only. The outlier rule uses fixed thresholds and only earlier readings, but the thresholds were chosen after looking at the full dataset (see the cleaning note above), so this item is met in the backward-looking sense, not the strict one.
8. The BA list is fixed by the rule above using 2019 to 2020 data only.
9. The API serves the latest revised values, not what was known in real time. This cannot be fixed, so it goes in the README.

## Results (cleaned data, 20 BAs, quarterly refits, test origins 2021-01-01 to 2026-11-01)

Produced by `scripts/run_evaluation.py` (outputs in the git-ignored `data/eval/`). Mean absolute percentage error over hourly forecasts, each model scored on the same rows. Variant A uses weather observed up to the origin; variant B uses observed weather for the target day and is an upper bound, not something a live system has.

| Model | A: all 20 BAs | B: all 20 BAs | A: 17 BAs with EIA forecast | B: 17 BAs with EIA forecast |
| --- | --- | --- | --- | --- |
| Seasonal-naive | 9.48 | 9.48 | 9.23 | 9.23 |
| Previous-day naive | 7.41 | 7.41 | 7.18 | 7.18 |
| Same-day-type average | 9.17 | 9.17 | 9.00 | 9.00 |
| Regression | 8.77 | 7.69 | 8.66 | 7.65 |
| LightGBM | 5.18 | 3.32 | 4.99 | 3.21 |
| EIA day-ahead forecast | n/a | n/a | 3.75 | 3.75 |

- LightGBM is best of the five models in both variants, in every year and in every BA. It is not the best against EIA's own forecast: with variant A weather it loses to it overall and in 13 of the 17 BAs; with perfect target-day weather (B) it wins overall and in 10 of 17.
- Error by regime on the rows where EIA's forecast exists (A / B for LightGBM): Uri 6.19 / 3.86 against EIA 5.55; Elliott 7.97 / 4.81 against 4.40; ERCOT summer 2023 5.31 / 3.19 against 2.04; holidays and adjacent days 5.45 / 3.59 against 3.96; calm days 4.93 / 3.17 against 3.73. In the Texas heat, previous-day naive (4.15) beats variant A LightGBM.
- Error rises in later years for LightGBM and for EIA's forecast alike. LightGBM in A: 4.89% in 2021 to 5.76% in 2026. EIA: 3.12% to 5.55%. The cause was not investigated.
- Regression is worse than previous-day naive in variant A and slightly better in B. Its poor showing in the Texas heat (about 11%) and in several BAs shows that the simple heating and cooling terms are too crude, so it is a weak benchmark and the LightGBM margin over it overstates the value of machine learning.
- Uri caveat as above: demand in those days is load served, so errors partly measure outages.
- **Forecast origin sensitivity** (`scripts/run_origin_sensitivity.py`, variant A, test origins 2024-01-01 onward, identical rows and models; the demand cutoff stays two hours before the origin and the weather cutoff one hour before). Seasonal-naive does not depend on the origin.

  | Origin | LightGBM, 20 BAs | Previous-day naive, 20 BAs | LightGBM, 17 BAs | Previous-day naive, 17 BAs | EIA forecast, 17 BAs |
  | --- | --- | --- | --- | --- | --- |
  | 06:00 | 5.77 | 7.96 | 5.57 | 7.71 | 4.33 |
  | 10:00 | 5.34 | 7.49 | 5.14 | 7.25 | 4.33 |
  | 12:00 | 5.10 | 7.26 | 4.90 | 7.01 | 4.33 |

  A later origin helps because the model sees more of the day's demand, and the gain is concentrated in the morning target hours (LightGBM 00:00 to 05:59 target error 4.63 / 4.13 / 3.96 and 06:00 to 11:59 5.54 / 5.07 / 4.72 at origins 06 / 10 / 12). The ranking of models and the finding that EIA's forecast beats variant A LightGBM do not change at any origin. 10:00 is kept as a middle option that sits at the time of day when day-ahead markets close (next bullet). A 12:00 origin would be after EIA's stated 11:00 ET publication time for every BA and after every day-ahead close below, so it is not a realistic day-ahead forecast.
- **Day-ahead timing evidence** (checked 2026-11-04 against the operators' own documents; the operators' filings with FERC are the regulatory record where noted):

  | Market (our BA) | Offers and bids close | Results | Source |
  | --- | --- | --- | --- |
  | NYISO (NYIS) | 5:00 a.m. ET for load forecasts and bids | Schedule posted by 11:00 a.m. | NYISO Market Services Tariff, sections 4.2.1.1 and 4.2.5 (copy on nyiso.com, undated) |
  | ISO-NE (ISNE) | 10:30 a.m. ET | After the market clears (the 2020 FERC filing says by 1:30 p.m.) | ISO-NE Manual M-11, revision 62, effective 2025-02-28, section 2.2.1; its history records the change from 10:00 in 2020 (the 10:00 deadline was accepted by FERC in 2013, 143 FERC 61,065) |
  | MISO (MISO) | 10:30 a.m. Eastern Prevailing Time | Not checked | MISO Tariff section 39.1.1, as reproduced in MISO's own tariff-revision draft of 2024-10-10; MISO's virtual trading onboarding document agrees |
  | PJM (PJM) | 11:00 a.m. ET | By 1:30 p.m. | PJM Manual 11, revision 137, effective 2026-07-28, sections 2.1 and 2.3.3A.1 (the 11:00 deadline was approved by FERC in 2018) |
  | ERCOT (ERCO) | 10:00 a.m. CT: DAM energy bids and offers "by 1000" | No later than 1:30 p.m. | ERCOT Nodal Protocols, Section 4 (Day-Ahead Operations), dated 2026-07-10 |
  | CAISO (CISO) | 10:00 a.m. PT | 1:00 p.m. | CAISO Business Practice Manual for Market Operations, v101.1, 2025-03-26 |
  | EIA (benchmark) | Each BA reports the day-ahead demand forecast it makes "in the normal course of business"; EIA states it is published by 11:00 ET | | Form EIA-930 instructions (OMB 1905-0129) |

  What this supports: the 10:00 to 11:00 morning window is where five of the six markets close, so 10:00 local is a representative time of day. What it does not support: that a forecast with a 10:00 information cutoff was feasible everywhere. It is exactly at the close for ERCOT and CAISO, 30 minutes before ISO-NE's close, later than NYISO's 5:00 ET, and for MISO (whose zone here is Central) 10:00 CT is 11:00 EPT, after the 10:30 EPT close. Only 6 of the 20 BAs sit in these markets. SWPP was not checked, and the other 13 are utility BAs for which no such deadline was found. EIA's forecast has no common timing, so its information set cannot be matched to ours exactly. If a BA builds its forecast before its market closes, EIA's information was no later than ours for ERCOT and CAISO, which would make our finding that EIA's forecast beats variant A LightGBM conservative; that is an inference, not a checked fact. Source caveats: PJM, ERCOT, ISO-NE and CAISO are cited to their current manuals or protocols. The MISO time is quoted from MISO's own draft of its tariff language, not the filed tariff or the current BPM-002 (a ZIP over 10 MB that the fetch tool could not read). The NYISO tariff text is a copy hosted on nyiso.com with no date on it; the current Day-Ahead Scheduling Manual could not be reached. Operators can extend a deadline after an outage, so these are scheduled times.

## Publishing rules

Under the EIA API terms, the dashboard may display the data but may not claim EIA as the source of modified content.

- **Unaltered EIA values** are credited "Source: U.S. Energy Information Administration".
- **Everything computed here** (cleaned series, forecasts, error metrics, SHAP values) is labeled "computed by the author from EIA data".
- **No EIA logo** and no wording that implies EIA endorsement.
- **Third-party material** on EIA's site (such as photos) is not used.
- **NOAA weather** is credited to NOAA/NCEI, and to the Iowa Environmental Mesonet for any dates taken from it.

## Known limitations for the README

These limits go in the README as stated facts, because naming them is part of what makes the project credible.

- **History:** the API starts in 2019, so the sample has few extreme events (Uri and Elliott are the main ones).
- **Revisions:** values are the latest revised ones, so backtests are easier than a live system's.
- **Imputation and cleaning:** EIA fills some anomalous demand values itself. The README states the added cleaning rule (0.4 to 2.0 times the previous 15 days' median), that its thresholds were chosen by eye on the full dataset, that it discards about 15 genuine extreme-event hours, and that a few dozen isolated dips remain.
- **Benchmark gaps:** EIA's day-ahead forecast is missing or unusable for FPC, PSCO and LGEE, so the comparison with it covers 17 of the 20 BAs.
- **Reporting quirks:** pseudo-ties and dynamic scheduling can distort a BA's demand. Distribution-connected generation such as rooftop solar is generally excluded.
- **Weather:** the main results use observed weather, not forecasts.
- **Scope:** conclusions are about the model versus baselines, not about outperforming operators.

## Work-checkpoint tasks

- [x] Confirm whether a period stamp marks the start or end of its hour (end)
- [x] Pull respondent metadata: which are BAs, aggregates or generation-only, and each BA's time zone
- [x] Map each selected BA to its civil time zone (`scripts/ba_timezones.py`; `scripts/timezone_check.py` confirms the daylight-saving shift for 18 of 20, TVA and NEVP show none and are treated as noise; MISO is Central and SOCO Eastern, unlike EIA's table)
- [x] Write the BA selection rule and run the completeness table (49 pass, 20 selected)
- [x] Download the day-ahead forecast (`DF`) for the same BAs (57 files in `data/raw/hourly/`)
- [x] Confirm the NOAA hourly access route and map stations to BAs (S3 bucket, no token; 42 stations downloaded for 2019 to 2025-08-27)
- [x] Extend weather past 2025-08-27 with the Iowa Mesonet (done; 42 stations to 2026-11-03)
- [x] Build the hourly weather series per BA (`scripts/weather_hourly.py` writes `data/weather_hourly.parquet`: UTC hour-ending stamps, equal-weight mean over stations, every BA has a temperature in at least 99.46% of hours). Check: in July and August 2023, ERCOT demand correlates 0.97 with temperature one hour earlier (PJM 0.92), with no peak at a negative lag
- [x] Check when EIA's day-ahead forecast is published, per BA (EIA states by 11:00 ET; the API carries no publication times, so per-BA timing can't be checked. The 10:00 local origin is before it for Eastern BAs and at or after it for the others, so it stays a benchmark only)
- [x] Write pseudocode for target and feature construction (section above)
- [x] Write the feature and target code with leakage tests (`electricity/features.py`, `tests/test_features.py`)
- [x] Build the models and the walk-forward evaluation (`electricity/models.py`, `electricity/evaluate.py`, `scripts/run_evaluation.py`)
- [x] Add the demand cleaning rule after the first evaluation showed impossible EIA values (`electricity/cleaning.py`, `tests/test_cleaning.py`; checks in `scripts/check_cleaning.py` and `scripts/run_sensitivity.py`)
- [x] Run the full evaluation for variants A and B on cleaned data (results section above)
- [x] Settle the forecast origin (10:00 local kept; 06:00 and 12:00 tested, see Results)
- [ ] SHAP importance for LightGBM
- [ ] Look into why error rises in 2025 and 2026
- [ ] Dashboard, with EIA values and computed values labeled separately
