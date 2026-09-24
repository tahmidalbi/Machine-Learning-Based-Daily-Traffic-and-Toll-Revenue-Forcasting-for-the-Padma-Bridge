# Padma Bridge Traffic & Toll Revenue Forecasting

A machine learning study of daily road traffic and toll revenue on Bangladesh's Padma Bridge. The repository brings together several related sub-projects that approach the same underlying data from different angles: one-day-ahead forecasting, multi-day-ahead forecasting, direction-specific (Mawa ↔ Jajira) forecasting, sequence-model time-series forecasting, and econometric impact analysis of the bridge's rail link and regional development effects.

## What the project answers

1. How accurately can next-day Padma Bridge traffic and toll revenue be forecast?
2. How far into the future (1–30 days) can traffic and toll be forecast, and how does accuracy degrade with horizon?
3. Do the two travel directions (Mawa-bound vs. Jajira-bound) behave differently enough to warrant separate models?
4. How do weather, weekends, public holidays, and Eid periods affect traffic?
5. Do changes in Padma and Jamuna bridge traffic contain predictive information about one another?
6. Did the start of regular commercial rail service through the Padma Bridge coincide with a change in road traffic?
7. Did nighttime-light intensity in southwestern districts change differently after the bridge opened?

## Repository layout

| Folder | What it does |
|---|---|
| [`Machine-Learning-Based-Daily-Traffic-and-Toll-Revenue-Forcasting-for-the-Padma-Bridge-main/`](./Machine-Learning-Based-Daily-Traffic-and-Toll-Revenue-Forcasting-for-the-Padma-Bridge-main) | The core project. An end-to-end, reproducible pipeline that forecasts **next-day** total traffic and toll revenue and runs the full econometric impact study (weather/holiday/Eid effects, Padma–Jamuna Granger analysis, railway intervention analysis, nighttime-light difference-in-differences). See its own README for full methodology, results tables, and usage. |
| [`padma_bridge_ml_complete/`](./padma_bridge_ml_complete) | A packaged/standalone copy of the core project (same pipeline, plus raw CSV inputs and a Colab notebook), intended as a self-contained, ready-to-run bundle — e.g. for uploading to Google Drive/Colab. |
| [`multihorizon_forecasting/`](./multihorizon_forecasting) | Extends the forecasting task from one day ahead to a **1–30 day horizon**, training global direct multi-horizon models (Random Forest, XGBoost, CatBoost, Residual MLP) with origin-date-safe features so no future information leaks into any horizon. |
| [`directional_forecasting/`](./directional_forecasting) | Splits total traffic/toll into **direction-specific** series (Mawa-bound and Jajira-bound) and trains separate models per direction and per target, to test whether directional detail improves on aggregate forecasts. |
| [`Padma_TimeSeries-master/`](./Padma_TimeSeries-master) | A **sequence-model** (PyTorch, windowed/raw time-series) approach to the same forecasting problem, independent of the tabular scikit-learn/XGBoost pipeline used elsewhere — includes architecture tuning, window-size sweeps, and multi-seed ensembling experiments. |
| `*.pdf` files at the root | Written report deliverables (project report and per-section "portion work" write-ups) that document the methodology and findings behind the directional and multi-horizon extensions. |

## Data

All sub-projects are built on the same underlying daily toll-report data for the Padma Bridge (traffic counts, toll revenue by vehicle class, and side/direction breakdowns), supplemented with:

- Daily weather (temperature, rainfall, humidity, wind)
- Public holiday and Eid calendars
- Jamuna Bridge traffic/toll history (a comparable crossing, used as a cross-corridor predictor)
- District-level nighttime-light intensity (for the regional development impact analysis)

Raw and processed CSVs live under each sub-project's own `data/` (or `raw/`) folder — see that sub-project's README for exact file names, coverage dates, and row counts.

## Models used across the project

- Naive baselines: previous-day persistence, seven-day seasonal persistence
- Linear Regression
- Random Forest
- XGBoost
- CatBoost (directional and multi-horizon variants)
- SARIMAX with exogenous variables
- A custom two-branch Residual MLP (PyTorch)
- Sequence models (PyTorch) for the raw time-series approach

## Getting started

Each sub-project is self-contained with its own `requirements.txt` and entry-point script(s). Start with the core project's README for the full picture, then explore the extensions:

```bash
# Core next-day forecasting + impact analysis pipeline
cd Machine-Learning-Based-Daily-Traffic-and-Toll-Revenue-Forcasting-for-the-Padma-Bridge-main
python -m venv .venv && .venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
python run_all.py --quick   # smoke test
python run_all.py           # full run
```

```bash
# Multi-horizon (1-30 day) forecasting
cd multihorizon_forecasting
python train_multihorizon.py
```

```bash
# Directional (Mawa vs. Jajira) forecasting
cd directional_forecasting
python train_directional.py
```

```bash
# Sequence-model (PyTorch) approach
cd Padma_TimeSeries-master
pip install -r requirements.txt
python train_model.py
```

## Reproducibility notes

- Every sub-project shifts traffic/toll/revenue history features by at least one day to avoid target leakage; only calendar and forecast-weather inputs are used as same-day features.
- Data is split chronologically (train → validation → test), never randomly.
- Randomized components use fixed seeds, and neural models are typically averaged across a multi-seed ensemble.
- Regression accuracy is reported via MAE, RMSE, MAPE, and R² — not classification accuracy.

## License and citation

No license or citation file is currently included at the repository level. Before public redistribution, add the intended software license and verify reuse/attribution terms for each source dataset (see the core project's README for data source links).
