from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import EXTERNAL_FEATURES, HISTORY_FEATURES, MODEL_FEATURES, RANDOM_SEED, SARIMAX_FEATURES, TARGETS
from .metrics import regression_metrics, segmented_metrics


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[train_end:validation_end].copy(), df.iloc[validation_end:].copy()


def _time_series_search(base_estimator, candidates, X, y, quick: bool = False):
    splits = 3 if quick else 5
    cv = TimeSeriesSplit(n_splits=splits)
    rows = []
    best_score = np.inf
    best_params = None
    for run, params in enumerate(candidates, start=1):
        fold_mae = []
        for train_idx, val_idx in cv.split(X):
            model = clone(base_estimator).set_params(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[val_idx])
            fold_mae.append(np.mean(np.abs(y.iloc[val_idx].to_numpy() - pred)))
        score = float(np.mean(fold_mae))
        rows.append({"run": run, "cv_mae": score, **params})
        if score < best_score:
            best_score, best_params = score, params
    return best_params, pd.DataFrame(rows).sort_values("cv_mae")


def _rf_candidates(quick: bool):
    space = {
        "n_estimators": [100, 300] if quick else [300, 500, 800, 1000],
        "max_features": [0.35, 0.55, 0.75, 1.0],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "max_depth": [None, 10, 15, 20, 25],
        "max_samples": [0.8, 0.9, 1.0],
    }
    return list(ParameterSampler(space, n_iter=4 if quick else 24, random_state=RANDOM_SEED))


def _xgb_candidates(quick: bool):
    space = {
        "n_estimators": [150, 300] if quick else [300, 500, 800, 1000],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.015, 0.03, 0.05, 0.08],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 8],
        "reg_lambda": [1.0, 3.0, 10.0],
    }
    return list(ParameterSampler(space, n_iter=3 if quick else 20, random_state=RANDOM_SEED))


def _prediction_frame(part: pd.DataFrame, target: str, model_name: str, values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": part["date"].to_numpy(),
            "actual": part[target].to_numpy(dtype=float),
            "predicted": np.maximum(np.asarray(values, dtype=float), 0),
            "model": model_name,
            "eid": part["eid"].to_numpy(),
            "is_holiday": part["is_holiday"].to_numpy(),
            "days_to_nearest_eid": part["days_to_nearest_eid"].to_numpy(),
        }
    )


def _new_residual_mlp(n_external: int, n_history: int):
    import torch
    import torch.nn as nn

    class ResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.external = nn.Sequential(
                nn.Linear(n_external, 32), nn.GELU(), nn.Dropout(0.12), nn.Linear(32, 16), nn.GELU()
            )
            self.history = nn.Sequential(
                nn.Linear(n_history, 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 32), nn.GELU()
            )
            self.fusion = nn.Sequential(
                nn.Linear(48, 32), nn.GELU(), nn.Dropout(0.12), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1)
            )

        def forward(self, external, history):
            return self.fusion(torch.cat([self.external(external), self.history(history)], dim=1))

    return ResidualMLP()


def _fit_mlp_seed(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target: str,
    base_feature: str,
    seed: int,
    quick: bool,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler().fit(train[MODEL_FEATURES])
    x_train = scaler.transform(train[MODEL_FEATURES]).astype("float32")
    x_val = scaler.transform(validation[MODEL_FEATURES]).astype("float32")
    residual = train[target].to_numpy(dtype="float32") - train[base_feature].to_numpy(dtype="float32")
    residual_mean = float(residual.mean())
    residual_std = float(residual.std() + 1e-6)
    residual_scaled = ((residual - residual_mean) / residual_std).astype("float32")

    n_ext = len(EXTERNAL_FEATURES)
    model = _new_residual_mlp(n_ext, len(HISTORY_FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = torch.nn.SmoothL1Loss()
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(residual_scaled[:, None]))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    max_epochs, patience = (80, 15) if quick else (500, 55)
    best_mae, best_epoch, best_state, stale = np.inf, 0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, rb in loader:
            xb, rb = xb.to(device), rb.to(device)
            optimizer.zero_grad()
            output = model(xb[:, :n_ext], xb[:, n_ext:])
            loss = criterion(output, rb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(x_val).to(device)
            delta = model(xv[:, :n_ext], xv[:, n_ext:]).cpu().numpy().ravel() * residual_std + residual_mean
        prediction = validation[base_feature].to_numpy(dtype=float) + delta
        mae = float(np.mean(np.abs(validation[target].to_numpy(dtype=float) - prediction)))
        if mae < best_mae - 1e-6:
            best_mae, best_epoch = mae, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return {
        "model": model,
        "scaler": scaler,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "best_epoch": best_epoch,
        "device": device,
    }


def _mlp_predict(bundle, frame: pd.DataFrame, base_feature: str) -> np.ndarray:
    import torch

    x = bundle["scaler"].transform(frame[MODEL_FEATURES]).astype("float32")
    n_ext = len(EXTERNAL_FEATURES)
    bundle["model"].eval()
    with torch.no_grad():
        xt = torch.from_numpy(x).to(bundle["device"])
        delta = bundle["model"](xt[:, :n_ext], xt[:, n_ext:]).cpu().numpy().ravel()
    delta = delta * bundle["residual_std"] + bundle["residual_mean"]
    return np.maximum(frame[base_feature].to_numpy(dtype=float) + delta, 0)


def _fit_mlp_fixed_epochs(frame, target, base_feature, seed, epochs):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler().fit(frame[MODEL_FEATURES])
    x = scaler.transform(frame[MODEL_FEATURES]).astype("float32")
    residual = frame[target].to_numpy(dtype="float32") - frame[base_feature].to_numpy(dtype="float32")
    residual_mean, residual_std = float(residual.mean()), float(residual.std() + 1e-6)
    r = ((residual - residual_mean) / residual_std).astype("float32")
    model = _new_residual_mlp(len(EXTERNAL_FEATURES), len(HISTORY_FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = torch.nn.SmoothL1Loss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(r[:, None])), batch_size=64, shuffle=True)
    model.train()
    for _ in range(max(1, int(epochs))):
        for xb, rb in loader:
            xb, rb = xb.to(device), rb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb[:, : len(EXTERNAL_FEATURES)], xb[:, len(EXTERNAL_FEATURES) :]), rb)
            loss.backward()
            optimizer.step()
    return {
        "model": model,
        "scaler": scaler,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "best_epoch": int(epochs),
        "device": device,
    }


def _save_mlp_ensemble(path: Path, bundles, target: str, base_feature: str):
    import torch

    payload = {
        "target": target,
        "base_feature": base_feature,
        "features": MODEL_FEATURES,
        "external_features": EXTERNAL_FEATURES,
        "history_features": HISTORY_FEATURES,
        "members": [
            {
                "state_dict": {k: v.detach().cpu() for k, v in b["model"].state_dict().items()},
                "scaler_mean": b["scaler"].mean_,
                "scaler_scale": b["scaler"].scale_,
                "residual_mean": b["residual_mean"],
                "residual_std": b["residual_std"],
                "epochs": b["best_epoch"],
            }
            for b in bundles
        ],
    }
    torch.save(payload, path)


def _sarimax_walk_forward(train, future, target, order, seasonal_order):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    scaler = StandardScaler().fit(train[SARIMAX_FEATURES])
    train_exog = scaler.transform(train[SARIMAX_FEATURES])
    future_exog = scaler.transform(future[SARIMAX_FEATURES])
    model = SARIMAX(
        train[target].to_numpy(dtype=float),
        exog=train_exog,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=200)
    predictions = []
    current = result
    for i, actual in enumerate(future[target].to_numpy(dtype=float)):
        exog_row = future_exog[i : i + 1]
        prediction = float(np.asarray(current.forecast(steps=1, exog=exog_row)).ravel()[0])
        predictions.append(max(prediction, 0.0))
        current = current.append(endog=np.asarray([actual]), exog=exog_row, refit=False)
    return np.asarray(predictions), current, scaler


def _run_sarimax(train, val, test, target, model_dir, quick):
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        warnings.warn("statsmodels is unavailable; skipping SARIMAX.")
        return None
    candidates = [
        ((1, 0, 1), (1, 0, 0, 7)),
        ((2, 0, 1), (1, 0, 0, 7)),
        ((1, 1, 1), (0, 0, 1, 7)),
        ((2, 1, 1), (1, 0, 0, 7)),
    ]
    if quick:
        candidates = candidates[:1]
    rows, best = [], None
    for order, seasonal in candidates:
        try:
            prediction, _, _ = _sarimax_walk_forward(train, val, target, order, seasonal)
            mae = regression_metrics(val[target], prediction)["MAE"]
            rows.append({"order": str(order), "seasonal_order": str(seasonal), "validation_mae": mae})
            if best is None or mae < best[0]:
                best = (mae, order, seasonal, prediction)
        except Exception as exc:
            rows.append({"order": str(order), "seasonal_order": str(seasonal), "validation_mae": np.nan, "error": str(exc)})
    if best is None:
        warnings.warn("All SARIMAX configurations failed; see run details.")
        return {"failed_candidates": rows}
    development = pd.concat([train, val], ignore_index=True)
    test_prediction, final_result, scaler = _sarimax_walk_forward(development, test, target, best[1], best[2])
    final_result.save(model_dir / f"{target}_sarimax.pkl")
    joblib.dump({"scaler": scaler, "features": SARIMAX_FEATURES, "order": best[1], "seasonal_order": best[2]}, model_dir / f"{target}_sarimax_metadata.joblib")
    return {
        "validation_prediction": best[3],
        "test_prediction": test_prediction,
        "best_order": best[1],
        "best_seasonal_order": best[2],
        "candidates": rows,
    }


def _run_custom_mlp(train, val, test, target, base_feature, model_dir, quick):
    try:
        import torch  # noqa: F401
    except ImportError:
        warnings.warn("PyTorch is unavailable; skipping the custom Residual MLP.")
        return None
    seeds = [1] if quick else [1, 7, 42]
    initial = [_fit_mlp_seed(train, val, target, base_feature, seed, quick) for seed in seeds]
    val_pred = np.mean([_mlp_predict(bundle, val, base_feature) for bundle in initial], axis=0)
    epochs = int(np.median([bundle["best_epoch"] for bundle in initial]))
    development = pd.concat([train, val], ignore_index=True)
    final = [_fit_mlp_fixed_epochs(development, target, base_feature, seed, epochs) for seed in seeds]
    test_pred = np.mean([_mlp_predict(bundle, test, base_feature) for bundle in final], axis=0)
    _save_mlp_ensemble(model_dir / f"{target}_residual_mlp.pt", final, target, base_feature)
    return val_pred, test_pred, {"seeds": seeds, "final_epochs": epochs}


def _plot_predictions(test_predictions: pd.DataFrame, target_label: str, output_path: Path):
    plt.figure(figsize=(14, 7))
    full_index = pd.date_range(test_predictions["date"].min(), test_predictions["date"].max(), freq="D")
    actual = test_predictions.drop_duplicates("date").set_index("date")["actual"].reindex(full_index)
    plt.plot(full_index, actual, color="black", linewidth=2.2, label="Actual")
    for model, group in test_predictions.groupby("model"):
        series = group.set_index("date")["predicted"].reindex(full_index)
        plt.plot(full_index, series, linewidth=1.1, alpha=0.82, label=model)
    plt.title(f"Padma Bridge {target_label}: actual vs one-day-ahead predictions")
    plt.xlabel("Date")
    plt.ylabel(target_label)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def train_target(
    modeling_table: Path,
    target_key: str,
    model_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    quick: bool = False,
) -> dict:
    target = TARGETS[target_key]
    base_feature = "padma_traffic_lag_1" if target_key == "traffic" else "padma_cash_lag_1"
    df = pd.read_csv(modeling_table, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=[target, *MODEL_FEATURES]).reset_index(drop=True)
    train, val, test = chronological_split(df)
    X_train, y_train = train[MODEL_FEATURES], train[target]
    X_val, X_test = val[MODEL_FEATURES], test[MODEL_FEATURES]
    development = pd.concat([train, val], ignore_index=True)

    validation_predictions = []
    test_predictions = []
    details = {}

    for name, feature in [("Persistence-1", base_feature), ("Seasonal-7", "padma_traffic_lag_7" if target_key == "traffic" else "padma_cash_lag_7")]:
        validation_predictions.append(_prediction_frame(val, target, name, val[feature]))
        test_predictions.append(_prediction_frame(test, target, name, test[feature]))

    linear = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    linear.fit(X_train, y_train)
    validation_predictions.append(_prediction_frame(val, target, "LinearRegression", linear.predict(X_val)))
    linear_final = clone(linear).fit(development[MODEL_FEATURES], development[target])
    test_predictions.append(_prediction_frame(test, target, "LinearRegression", linear_final.predict(X_test)))
    joblib.dump({"model": linear_final, "features": MODEL_FEATURES, "target": target}, model_dir / f"{target}_linear.joblib")

    rf_base = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
    rf_params, rf_search = _time_series_search(rf_base, _rf_candidates(quick), X_train, y_train, quick)
    rf_search.to_csv(tables_dir / f"{target_key}_rf_tuning.csv", index=False)
    rf = clone(rf_base).set_params(**rf_params).fit(X_train, y_train)
    validation_predictions.append(_prediction_frame(val, target, "RandomForest", rf.predict(X_val)))
    rf_final = clone(rf_base).set_params(**rf_params).fit(development[MODEL_FEATURES], development[target])
    test_predictions.append(_prediction_frame(test, target, "RandomForest", rf_final.predict(X_test)))
    joblib.dump({"model": rf_final, "features": MODEL_FEATURES, "target": target}, model_dir / f"{target}_random_forest.joblib")
    details["random_forest_best_params"] = rf_params

    try:
        from xgboost import XGBRegressor

        xgb_base = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            tree_method="hist",
            n_jobs=-1,
        )
        xgb_params, xgb_search = _time_series_search(xgb_base, _xgb_candidates(quick), X_train, y_train, quick)
        xgb_search.to_csv(tables_dir / f"{target_key}_xgb_tuning.csv", index=False)
        xgb = clone(xgb_base).set_params(**xgb_params).fit(X_train, y_train)
        validation_predictions.append(_prediction_frame(val, target, "XGBoost", xgb.predict(X_val)))
        xgb_final = clone(xgb_base).set_params(**xgb_params).fit(development[MODEL_FEATURES], development[target])
        test_predictions.append(_prediction_frame(test, target, "XGBoost", xgb_final.predict(X_test)))
        joblib.dump({"model": xgb_final, "features": MODEL_FEATURES, "target": target}, model_dir / f"{target}_xgboost.joblib")
        details["xgboost_best_params"] = xgb_params
    except ImportError:
        warnings.warn("xgboost is unavailable; install requirements.txt to include XGBoost.")
        details["xgboost"] = "skipped: package unavailable"

    mlp = _run_custom_mlp(train, val, test, target, base_feature, model_dir, quick)
    if mlp is not None:
        val_pred, test_pred, mlp_details = mlp
        validation_predictions.append(_prediction_frame(val, target, "ResidualMLP", val_pred))
        test_predictions.append(_prediction_frame(test, target, "ResidualMLP", test_pred))
        details["residual_mlp"] = mlp_details

    sarimax = _run_sarimax(train, val, test, target, model_dir, quick)
    if sarimax is not None and "validation_prediction" in sarimax:
        validation_predictions.append(_prediction_frame(val, target, "SARIMAX", sarimax["validation_prediction"]))
        test_predictions.append(_prediction_frame(test, target, "SARIMAX", sarimax["test_prediction"]))
        details["sarimax"] = {
            "best_order": sarimax["best_order"],
            "best_seasonal_order": sarimax["best_seasonal_order"],
            "candidates": sarimax["candidates"],
        }
    elif sarimax is not None:
        details["sarimax"] = sarimax

    validation_predictions = pd.concat(validation_predictions, ignore_index=True)
    test_predictions = pd.concat(test_predictions, ignore_index=True)
    validation_predictions.to_csv(tables_dir / f"{target_key}_validation_predictions.csv", index=False)
    test_predictions.to_csv(tables_dir / f"{target_key}_test_predictions.csv", index=False)

    validation_metrics = segmented_metrics(validation_predictions)
    test_metrics = segmented_metrics(test_predictions)
    validation_metrics.to_csv(tables_dir / f"{target_key}_validation_metrics.csv", index=False)
    test_metrics.to_csv(tables_dir / f"{target_key}_test_metrics.csv", index=False)
    _plot_predictions(test_predictions, "traffic (vehicles)" if target_key == "traffic" else "toll revenue (BDT)", figures_dir / f"{target_key}_actual_vs_predicted.png")

    details["split"] = {
        "train": [str(train.date.min().date()), str(train.date.max().date()), int(len(train))],
        "validation": [str(val.date.min().date()), str(val.date.max().date()), int(len(val))],
        "test": [str(test.date.min().date()), str(test.date.max().date()), int(len(test))],
    }
    details["validation_overall"] = validation_metrics.query("segment == 'overall'").to_dict("records")
    details["test_overall"] = test_metrics.query("segment == 'overall'").to_dict("records")
    (tables_dir / f"{target_key}_run_details.json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
    return details
