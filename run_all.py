from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from src.config import ensure_directories
from src.data_pipeline import build_datasets
from src.explainability import analyze_weather_holidays, explain_random_forest, feature_ablation
from src.forecasting import train_target
from src.impact_analysis import nighttime_light_did, railway_interrupted_time_series
from src.time_series_analysis import analyze_padma_jamuna


def safe_run(name, function, *args, **kwargs):
    print(f"\n===== {name} =====", flush=True)
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        print(f"{name} failed: {exc}")
        traceback.print_exc()
        return {"status": "failed", "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Complete Padma Bridge ML and impact-analysis pipeline")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing the three required CSV files")
    parser.add_argument("--quick", action="store_true", help="Small smoke-test search; use full mode for final results")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = args.data_dir.resolve() if args.data_dir else project_root / "data" / "raw"
    paths = ensure_directories(project_root)

    summary = {}
    summary["data_audit"] = safe_run("1. Data cleaning and feature engineering", build_datasets, data_dir, paths["processed"], paths["reports"])
    modeling_table = paths["processed"] / "forecast_modeling_table.csv"
    daily_path = paths["processed"] / "merged_daily_with_features.csv"
    ntl_path = paths["processed"] / "ntl_panel.csv"

    if not modeling_table.exists():
        raise RuntimeError("Dataset construction failed; forecasting cannot continue.")

    target_steps = {"traffic": (2, 3, 4), "toll": (5, 6, 7)}
    for target_key in ["traffic", "toll"]:
        forecast_step, explain_step, ablation_step = target_steps[target_key]
        summary[f"forecast_{target_key}"] = safe_run(
            f"{forecast_step}. Forecasting: {target_key}",
            train_target,
            modeling_table,
            target_key,
            paths["models"],
            paths["tables"],
            paths["figures"],
            args.quick,
        )
        safe_run(
            f"{explain_step}. Explainability: {target_key}",
            explain_random_forest,
            modeling_table,
            target_key,
            paths["models"],
            paths["tables"],
            paths["figures"],
        )
        safe_run(
            f"{ablation_step}. Feature ablation: {target_key}",
            feature_ablation,
            modeling_table,
            target_key,
            paths["tables"],
            args.quick,
        )

    safe_run("8. Weather, holiday and Eid analysis", analyze_weather_holidays, daily_path, paths["tables"], paths["figures"])
    summary["padma_jamuna"] = safe_run(
        "9. Padma-Jamuna dynamic analysis", analyze_padma_jamuna, daily_path, paths["tables"], paths["figures"], paths["reports"]
    )
    summary["railway_its"] = safe_run(
        "10. Railway interrupted time-series analysis",
        railway_interrupted_time_series,
        daily_path,
        paths["tables"],
        paths["figures"],
        paths["reports"],
    )
    summary["ntl_did"] = safe_run(
        "11. Nighttime-light Difference-in-Differences",
        nighttime_light_did,
        ntl_path,
        paths["tables"],
        paths["figures"],
        paths["reports"],
    )
    (paths["outputs"] / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nComplete. Results are in: {paths['outputs']}")


if __name__ == "__main__":
    main()
