from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE_pct": np.nan, "R2": np.nan}
    nonzero = np.abs(y_true) > 1e-12
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_pct": float(mape),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def segmented_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    masks = {
        "overall": pd.Series(True, index=predictions.index),
        "normal_days": (predictions["eid"] == 0) & (predictions["is_holiday"] == 0),
        "public_holidays": predictions["is_holiday"] == 1,
        "eid_window_7d": predictions["days_to_nearest_eid"] <= 7,
    }
    for model in sorted(predictions["model"].unique()):
        model_rows = predictions["model"] == model
        for segment, mask in masks.items():
            subset = predictions.loc[model_rows & mask]
            result = regression_metrics(subset["actual"], subset["predicted"])
            rows.append({"model": model, "segment": segment, **result})
    return pd.DataFrame(rows)

