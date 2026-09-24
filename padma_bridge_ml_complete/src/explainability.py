from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

from .config import EXTERNAL_FEATURES, HISTORY_FEATURES, MODEL_FEATURES, TARGETS
from .forecasting import chronological_split
from .metrics import regression_metrics


def explain_random_forest(modeling_table: Path, target_key: str, model_dir: Path, tables_dir: Path, figures_dir: Path):
    target = TARGETS[target_key]
    df = pd.read_csv(modeling_table, parse_dates=["date"]).dropna(subset=[target, *MODEL_FEATURES])
    _, _, test = chronological_split(df)
    bundle = joblib.load(model_dir / f"{target}_random_forest.joblib")
    model = bundle["model"]

    native = pd.DataFrame({"feature": MODEL_FEATURES, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    native.to_csv(tables_dir / f"{target_key}_rf_impurity_importance.csv", index=False)

    permutation = permutation_importance(
        model,
        test[MODEL_FEATURES],
        test[target],
        scoring="neg_mean_absolute_error",
        n_repeats=15,
        random_state=42,
        n_jobs=-1,
    )
    perm = pd.DataFrame(
        {"feature": MODEL_FEATURES, "importance_mean": permutation.importances_mean, "importance_std": permutation.importances_std}
    ).sort_values("importance_mean", ascending=False)
    perm.to_csv(tables_dir / f"{target_key}_rf_permutation_importance.csv", index=False)

    top = perm.head(18).sort_values("importance_mean")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#2673b8", alpha=0.9)
    plt.xlabel("Increase in MAE after permutation")
    plt.title(f"Random Forest permutation importance: {target_key}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{target_key}_rf_permutation_importance.png", dpi=180)
    plt.close()

    try:
        import shap

        sample = test.sample(min(500, len(test)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample[MODEL_FEATURES], check_additivity=False)
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.title(f"SHAP summary: {target_key}")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{target_key}_rf_shap_summary.png", dpi=180, bbox_inches="tight")
        plt.close()
    except ImportError:
        warnings.warn("shap is unavailable; permutation and impurity importance were still generated.")


def analyze_weather_holidays(daily_path: Path, tables_dir: Path, figures_dir: Path):
    df = pd.read_csv(daily_path, parse_dates=["date"])
    df = df.dropna(subset=["padma_total_traffic"]).copy()
    weather = ["temp_mean_c", "temp_max_c", "temp_min_c", "rainfall_mm", "humidity_pct", "wind_speed_kmh"]
    corr = df[["padma_total_traffic", *weather]].corr(method="spearman")
    corr.to_csv(tables_dir / "weather_spearman_correlations.csv")

    rows = []
    for variable in ["rainy_day", "heavy_rain", "weekend", "is_holiday", "eid"]:
        for value, group in df.groupby(variable):
            rows.append(
                {
                    "variable": variable,
                    "value": int(value),
                    "n": int(len(group)),
                    "mean_traffic": float(group["padma_total_traffic"].mean()),
                    "median_traffic": float(group["padma_total_traffic"].median()),
                    "std_traffic": float(group["padma_total_traffic"].std()),
                }
            )
    pd.DataFrame(rows).to_csv(tables_dir / "weather_holiday_group_comparison.csv", index=False)

    event = (
        df.loc[df["eid_relative_day"].between(-14, 14)]
        .groupby("eid_relative_day", as_index=False)["padma_total_traffic"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    event.to_csv(tables_dir / "eid_event_profile.csv", index=False)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=event, x="eid_relative_day", y="mean", marker="o", color="#b23a48")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Days relative to nearest Eid (negative = before)")
    plt.ylabel("Mean Padma traffic")
    plt.title("Average Padma traffic around Eid")
    plt.tight_layout()
    plt.savefig(figures_dir / "eid_event_profile.png", dpi=180)
    plt.close()


def feature_ablation(modeling_table: Path, target_key: str, tables_dir: Path, quick: bool = False):
    target = TARGETS[target_key]
    df = pd.read_csv(modeling_table, parse_dates=["date"]).dropna(subset=[target, *MODEL_FEATURES])
    train, val, test = chronological_split(df)
    development = pd.concat([train, val], ignore_index=True)
    weather = [
        "temp_mean_c", "temp_max_c", "temp_min_c", "rainfall_mm", "humidity_pct", "wind_speed_kmh", "rainy_day", "heavy_rain"
    ]
    calendar = [feature for feature in EXTERNAL_FEATURES if feature not in weather]
    padma_traffic = [feature for feature in HISTORY_FEATURES if feature.startswith("padma_traffic") or feature.startswith("padma_roll")]
    jamuna = [feature for feature in HISTORY_FEATURES if feature.startswith("jamuna_")]
    toll = [feature for feature in HISTORY_FEATURES if "cash" in feature or feature.startswith("revenue_per_vehicle")]
    sets = {
        "A_weather_only": weather,
        "B_weather_calendar": weather + calendar,
        "C_add_padma_history": weather + calendar + padma_traffic,
        "D_add_jamuna_history": weather + calendar + padma_traffic + jamuna,
        "E_full_add_toll_history": list(dict.fromkeys(weather + calendar + padma_traffic + jamuna + toll)),
    }
    rows = []
    for name, features in sets.items():
        model = RandomForestRegressor(
            n_estimators=150 if quick else 600,
            max_features=0.7,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ).fit(development[features], development[target])
        metrics = regression_metrics(test[target], model.predict(test[features]))
        rows.append({"feature_set": name, "feature_count": len(features), **metrics})
    pd.DataFrame(rows).to_csv(tables_dir / f"{target_key}_feature_ablation.csv", index=False)
