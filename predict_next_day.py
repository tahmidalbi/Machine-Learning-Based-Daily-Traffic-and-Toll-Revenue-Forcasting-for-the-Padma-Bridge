from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES
from src.data_pipeline import add_features


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the next single day after the latest processed Padma observation")
    parser.add_argument("--date", required=True, help="Prediction date, YYYY-MM-DD; must be one day after the latest observed date")
    parser.add_argument("--temp-mean", type=float, required=True)
    parser.add_argument("--temp-max", type=float, required=True)
    parser.add_argument("--temp-min", type=float, required=True)
    parser.add_argument("--rainfall", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--wind-speed", type=float, required=True)
    parser.add_argument("--is-holiday", type=int, choices=[0, 1], default=0)
    parser.add_argument("--eid", type=int, choices=[0, 1], default=0)
    parser.add_argument("--days-to-eid", type=int, required=True)
    parser.add_argument("--eid-relative-day", type=int, required=True, help="Negative before Eid, zero on Eid, positive after Eid")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main():
    args = parse_args()
    project = args.project_dir.resolve()
    daily = pd.read_csv(project / "data" / "processed" / "merged_daily_with_features.csv", parse_dates=["date"])
    prediction_date = pd.Timestamp(args.date)
    latest_observed = daily.loc[daily["padma_total_traffic"].notna(), "date"].max()
    if prediction_date != latest_observed + pd.Timedelta(days=1):
        raise ValueError(
            f"Prediction date must be {str((latest_observed + pd.Timedelta(days=1)).date())}. "
            "Append real intervening observations and rerun run_all.py before forecasting a later date."
        )

    future = {column: np.nan for column in daily.columns}
    future.update(
        {
            "date": prediction_date,
            "temp_mean_c": args.temp_mean,
            "temp_max_c": args.temp_max,
            "temp_min_c": args.temp_min,
            "rainfall_mm": args.rainfall,
            "humidity_pct": args.humidity,
            "wind_speed_kmh": args.wind_speed,
            "is_holiday": args.is_holiday,
            "eid": args.eid,
            "days_to_nearest_eid": args.days_to_eid,
            "weekend": int(prediction_date.dayofweek in [4, 5]),
        }
    )
    base_columns = [
        "date", "padma_total_traffic", "padma_total_cash", "jamuna_total_traffic", "jamuna_total_cash",
        "temp_mean_c", "temp_max_c", "temp_min_c", "rainfall_mm", "humidity_pct", "wind_speed_kmh",
        "weekend", "is_holiday", "eid", "days_to_nearest_eid",
    ]
    history = daily[base_columns].copy()
    extended = pd.concat([history, pd.DataFrame([{key: future.get(key, np.nan) for key in base_columns}])], ignore_index=True)
    eid_dates = pd.DatetimeIndex(history.loc[history["days_to_nearest_eid"] == 0, "date"].dropna().unique())
    featured = add_features(extended, eid_dates=eid_dates)
    row = featured.iloc[[-1]].copy()
    row["eid_relative_day"] = args.eid_relative_day
    missing = [feature for feature in MODEL_FEATURES if pd.isna(row.iloc[0][feature])]
    if missing:
        raise ValueError(f"Cannot predict because these engineered features are missing: {missing}")

    outputs = {}
    for key, target in [("traffic", "padma_total_traffic"), ("toll", "padma_total_cash")]:
        bundle = joblib.load(project / "models" / f"{target}_random_forest.joblib")
        outputs[key] = max(float(bundle["model"].predict(row[bundle["features"]])[0]), 0)
    print(f"Prediction date: {prediction_date.date()}")
    print(f"Predicted traffic: {outputs['traffic']:,.0f} vehicles")
    print(f"Predicted toll revenue: BDT {outputs['toll']:,.0f}")
    print("Note: weather inputs should be forecasts available before the prediction date.")


if __name__ == "__main__":
    main()

