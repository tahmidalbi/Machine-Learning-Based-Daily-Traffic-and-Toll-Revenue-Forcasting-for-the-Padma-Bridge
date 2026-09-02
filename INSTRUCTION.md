# Padma Bridge Traffic, Toll and Impact Analysis

This is the complete Python/Google Colab implementation for the project brief
`Machine Learning-Based Daily Traffic and Toll Revenue Forecasting for the Padma Bridge`.

It implements all five parts promised in the brief:

1. Daily Padma traffic and toll forecasting
2. Weather, holiday and Eid effect analysis
3. Padma-Jamuna cross-correlation, Granger tests, VAR and impulse responses
4. Railway-opening interrupted time-series analysis
5. Nighttime-light Difference-in-Differences analysis

The forecasting comparison includes persistence baselines, Linear Regression,
Random Forest, XGBoost, SARIMAX and a custom two-branch Residual MLP.

## 1. Use only these three CSV files

Place these files inside `data/raw/`:

```text
padma_toll_report_with_holidays_weather.csv
jamuna_toll_report.csv
bangladesh_ntl_2019_2025_padma_groups.csv
```

Do not concatenate all supplied CSVs. The other files are subsets/intermediate
versions and would duplicate the same observations.

## 2. Important audit findings already handled

- Padma has 1,520 rows from 2022-06-26 to 2026-08-24, with 2026-03-31 absent.
- Jamuna has 1,515 rows from 2022-07-01 to 2026-08-24, with 2025-09-17 absent.
- NTL has 5,376 rows: 64 districts x 84 months from 2019-01 through 2025-12.
- Bridge numeric fields use Indian comma grouping and contain malformed punctuation.
  The parser handles examples such as `1,08,87,150.00`, `6.337`, and
  `90.65,950.00`.
- Some reported side values do not equal the reported total. The code preserves
  the published total used as the target and records every mismatch in
  `outputs/reports/data_audit.json`. It never silently invents a correction.
- Every traffic, toll and Jamuna lag/rolling feature is shifted by at least one
  day. The target day's Mawa/Jajira traffic and toll are never predictors.

## 3. Google Colab: exact steps

### Step 1 - Upload the project

Unzip `padma_bridge_ml_complete.zip` on your computer. Upload the resulting
`padma_bridge_ml_complete` folder to Google Drive. Put the three required CSVs
in its `data/raw/` directory.

Recommended Drive location:

```text
MyDrive/padma_bridge_ml_complete/
```

### Step 2 - Open a new Colab notebook

Select:

```text
Runtime -> Change runtime type -> T4 GPU
```

The GPU is only useful for the custom MLP. Random Forest and the statistical
models are small enough for CPU.

### Step 3 - Mount Drive

Copy into the first Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4 - Enter the project folder

```python
%cd /content/drive/MyDrive/padma_bridge_ml_complete
```

### Step 5 - Install exact dependencies

```python
!pip -q install -r requirements.txt
```

If Colab asks for a runtime restart after installation, restart it, remount
Drive, and run the `%cd` cell again.

### Step 6 - Run a fast smoke test

```python
!python run_all.py --quick
```

Expected quick-run behavior: all eleven numbered stages should finish without
an `analysis failed` traceback. Statistical-model convergence warnings, if any,
must be inspected, but ordinary package deprecation/index warnings are suppressed.

Then run the explicit leakage/parser checks:

```python
!python tests/test_data_pipeline.py
```

This checks file names, parsing, features and every available analysis with a
small hyperparameter search. Do not report the quick-run results as final.

### Step 7 - Run the final experiment

```python
!python run_all.py
```

The full run performs chronological tuning, trains final models and writes all
tables/figures/models.

## 4. What each code file does

```text
run_all.py
    Runs the entire project in the correct order.

predict_next_day.py
    Uses the saved final Random Forest models for a genuine next-day forecast.

src/config.py
    Exact source names, intervention dates, feature lists and output folders.

src/data_pipeline.py
    Numeric cleaning, date audit, daily alignment, leakage-safe lags/rolling
    statistics, calendar/Eid/weather/railway features and processed CSV export.

src/metrics.py
    MAE, RMSE, MAPE and R2, including normal/holiday/Eid segments.

src/forecasting.py
    Chronological 70/15/15 split; persistence baselines; Linear Regression;
    time-series cross-validated Random Forest and XGBoost tuning; walk-forward
    SARIMAX; custom Residual MLP; final refit and test evaluation.

src/explainability.py
    Impurity importance, permutation importance, SHAP, feature ablations,
    weather comparisons and the signed Eid event profile.

src/time_series_analysis.py
    ADF tests, log-first-difference cross-correlation, bidirectional Granger
    tests, VAR lag selection and 14-day impulse responses.

src/impact_analysis.py
    Railway interrupted time series with HAC standard errors and nighttime-light
    DiD with district/month fixed effects and district-clustered standard errors.
```

## 5. Forecasting design

Each complete modeling row predicts the current day's target using information
available before or for that day:

```text
known calendar + forecast weather + prior Padma/Jamuna history -> target day
```

The split is strictly chronological:

```text
first 70%       training
next 15%        validation
last 15%        untouched final test
```

Within the training period, Random Forest and XGBoost hyperparameters are
selected with expanding-window `TimeSeriesSplit`. Validation compares model
families. Each chosen configuration is then refitted on train+validation and
evaluated once on the test period.

The test protocol is one-day-ahead rolling evaluation. A row may use the real
traffic observed on previous test days, exactly as a deployed system would use
yesterday's completed bridge record to predict today.

### Random Forest search

The full run samples 24 configurations across:

```text
trees:             300, 500, 800, 1000
max features:      0.35, 0.55, 0.75, 1.00
minimum leaf size: 1, 2, 4, 6, 8
maximum depth:     unlimited, 10, 15, 20, 25
row sample:        0.80, 0.90, 1.00
```

The selection score is mean validation MAE across chronological folds.

### Custom Residual MLP

The custom network predicts change relative to yesterday:

```text
predicted target(t) = target(t-1) + predicted change(t)
```

Its two branches are:

```text
External branch: calendar, Eid, weather, railway -> 32 -> 16
History branch: Padma/Jamuna lags and rolling values -> 64 -> 32
Concatenate -> 32 -> 16 -> residual output
```

It uses GELU, dropout, AdamW, Huber loss, early stopping and a three-seed
ensemble. After validation finds the appropriate epoch count, the ensemble is
retrained on train+validation.

## 6. Output files to use in the report

### Main forecasting tables

```text
outputs/tables/traffic_test_metrics.csv
outputs/tables/toll_test_metrics.csv
```

Use rows where `segment = overall` for the main comparison table. Lower MAE,
RMSE and MAPE are better; higher R2 is better.

Regression does not have a classification-style accuracy percentage. If a
single percentage is required, report:

```text
approximate forecasting accuracy = 100 - test MAPE
```

Label it exactly as an MAPE-derived approximation, not ordinary accuracy.

### High-variation performance

The same files contain:

```text
normal_days
public_holidays
eid_window_7d
```

This tests whether a model that is good on average still handles demand shocks.

### Model predictions and plots

```text
outputs/tables/traffic_test_predictions.csv
outputs/tables/toll_test_predictions.csv
outputs/figures/traffic_actual_vs_predicted.png
outputs/figures/toll_actual_vs_predicted.png
```

### Explainability and ablation

```text
outputs/tables/traffic_rf_permutation_importance.csv
outputs/figures/traffic_rf_shap_summary.png
outputs/tables/traffic_feature_ablation.csv
outputs/tables/weather_holiday_group_comparison.csv
outputs/tables/eid_event_profile.csv
outputs/figures/eid_event_profile.png
```

Use permutation importance and SHAP together. Impurity importance alone can be
biased and correlated lag variables can share importance.

### Padma-Jamuna analysis

```text
outputs/tables/padma_jamuna_adf_tests.csv
outputs/tables/padma_jamuna_cross_correlation.csv
outputs/tables/padma_jamuna_granger.csv
outputs/reports/padma_jamuna_var_summary.txt
outputs/figures/padma_jamuna_impulse_response.png
```

The analysis uses log first differences, not raw trending levels, to reduce the
risk of spurious correlation. A small Granger p-value means useful past
predictive information; it does not prove physical causation.

### Railway intervention

```text
outputs/tables/railway_its_coefficients.csv
outputs/reports/railway_its_summary.json
outputs/figures/railway_intervention_monthly_traffic.png
```

The intervention is 2023-11-01, the beginning of regular commercial passenger
operation through the Padma Bridge, as reported by Bangladesh Sangbad Sangstha:
https://www.bssnews.net/news-flash/153019. The code separates an immediate level
change from an additional post-intervention trend and uses weekly HAC standard
errors. Because no daily train passenger counts or control corridor are used,
describe the estimates as adjusted associations, not definite causal effects.

### Nighttime-light DiD

```text
outputs/tables/ntl_before_after_groups.csv
outputs/reports/ntl_did_summary.json
outputs/reports/ntl_pretrend_model_summary.txt
outputs/figures/ntl_treatment_control_trends.png
```

The main regression uses district fixed effects, month fixed effects and
district-clustered standard errors. Check `parallel_trends_warning` before
making an economic-impact claim. If it is true, present the result as
descriptive/associational and discuss the violated pre-trend condition.

## 7. How to choose the final forecasting model

Do not assume the custom model or Random Forest wins. Use this order:

1. Reject any model that fails badly on validation or gives unstable outputs.
2. Compare final test MAE first, RMSE second, MAPE third and R2 fourth.
3. Check `eid_window_7d`; a tiny overall improvement is not worth a major Eid
   failure.
4. Confirm it beats both `Persistence-1` and `Seasonal-7`.
5. If two models are practically tied, prefer Random Forest because it is easier
   to explain and deploy with this small dataset.

## 8. Next-day prediction after training

The script permits only the day immediately after the latest observed Padma
record. This prevents accidentally forecasting multiple days while pretending
that unknown lag values are known.

Example for 2026-08-25 (replace weather/calendar inputs with verified forecasts):

```bash
python predict_next_day.py \
  --date 2026-08-25 \
  --temp-mean 29.2 \
  --temp-max 33.1 \
  --temp-min 26.4 \
  --rainfall 4.5 \
  --humidity 82 \
  --wind-speed 11.0 \
  --is-holiday 0 \
  --eid 0 \
  --days-to-eid 88 \
  --eid-relative-day 88
```

The numerical values above are only syntax examples, not real forecasts.

## 9. Minimum tables/figures for the final report

Include these:

1. Data audit and chronological split table
2. Traffic model comparison
3. Toll model comparison
4. Normal-day vs Eid-window error comparison
5. Actual vs predicted traffic plot
6. RF permutation importance and SHAP summary
7. Feature-ablation table
8. Eid-relative traffic plot
9. Padma-Jamuna Granger table and impulse response plot
10. Railway ITS coefficient table and monthly plot
11. NTL four-group table, DiD coefficient and pre-trend plot

## 10. Reproducibility

All randomized components use fixed seeds. Keep the generated
`outputs/run_summary.json`, the tuning CSVs, and the saved `models/` directory
with the submission. Never tune again after reading the final test results.

## 11. Source references from the project brief

```text
Bangladesh Bridge Authority: https://bba.gov.bd/
Open-Meteo historical weather: https://open-meteo.com/en/docs/historical-weather-api
Padma Bridge Rail Link Project: https://pbrlp.gov.bd/
World Bank Space2Stats NTL catalog:
https://datacatalog.worldbank.org/search/dataset/0066940/space2stats-monthly-annual-black-marble-nighttime-lights
```
