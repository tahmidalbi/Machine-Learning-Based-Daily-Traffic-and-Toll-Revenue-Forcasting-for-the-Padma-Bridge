from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import RAIL_COMMERCIAL_START


def railway_interrupted_time_series(daily_path: Path, tables_dir: Path, figures_dir: Path, reports_dir: Path):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        warnings.warn("statsmodels unavailable; railway ITS skipped.")
        return {"status": "skipped: statsmodels unavailable"}

    df = pd.read_csv(daily_path, parse_dates=["date"]).dropna(subset=["padma_total_traffic"]).copy()
    df["log_traffic"] = np.log1p(df["padma_total_traffic"])
    df["log_traffic_lag1"] = df["log_traffic"].shift(1)
    df["year_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["year_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)

    controls = "weekend + is_holiday + eid + rainfall_mm + temp_mean_c + humidity_pct + wind_speed_kmh + dow_sin + dow_cos + year_sin + year_cos"
    formulas = {
        "adjusted_its": f"log_traffic ~ time_index + rail_open + time_after_rail + {controls}",
        "lag_adjusted_sensitivity": f"log_traffic ~ time_index + rail_open + time_after_rail + log_traffic_lag1 + {controls}",
    }
    coefficient_rows, summaries = [], {}
    for name, formula in formulas.items():
        fitted = smf.ols(formula, data=df).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
        summaries[name] = fitted.summary().as_text()
        conf = fitted.conf_int()
        for term in fitted.params.index:
            coefficient_rows.append(
                {
                    "model": name,
                    "term": term,
                    "coefficient": fitted.params[term],
                    "std_error_HAC": fitted.bse[term],
                    "p_value": fitted.pvalues[term],
                    "ci_low": conf.loc[term, 0],
                    "ci_high": conf.loc[term, 1],
                }
            )
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients.to_csv(tables_dir / "railway_its_coefficients.csv", index=False)
    (reports_dir / "railway_its_model_summaries.txt").write_text("\n\n".join(f"=== {k} ===\n{v}" for k, v in summaries.items()), encoding="utf-8")

    main = coefficients.query("model == 'adjusted_its'").set_index("term")
    immediate = float((np.exp(main.loc["rail_open", "coefficient"]) - 1) * 100)
    slope = float((np.exp(main.loc["time_after_rail", "coefficient"]) - 1) * 100)
    summary = {
        "commercial_service_start": RAIL_COMMERCIAL_START,
        "estimated_immediate_percent_change": immediate,
        "estimated_additional_daily_trend_percent": slope,
        "immediate_change_p_value": float(main.loc["rail_open", "p_value"]),
        "trend_change_p_value": float(main.loc["time_after_rail", "p_value"]),
        "interpretation_limit": "This is an adjusted temporal association, not proof that rail service caused the change.",
    }
    (reports_dir / "railway_its_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    monthly = df.set_index("date")["padma_total_traffic"].resample("MS").mean().reset_index()
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=monthly, x="date", y="padma_total_traffic", color="#24557a")
    plt.axvline(pd.Timestamp(RAIL_COMMERCIAL_START), color="#b23a48", linestyle="--", label="Commercial rail service")
    plt.ylabel("Monthly mean daily road traffic")
    plt.title("Padma road traffic around rail-service introduction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "railway_intervention_monthly_traffic.png", dpi=180)
    plt.close()
    return summary


def nighttime_light_did(ntl_path: Path, tables_dir: Path, figures_dir: Path, reports_dir: Path):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        warnings.warn("statsmodels unavailable; nighttime-light DiD skipped.")
        return {"status": "skipped: statsmodels unavailable"}

    df = pd.read_csv(ntl_path, parse_dates=["date"]).copy()
    df["district"] = df["NAM_2"].astype(str)
    df["month_id"] = df["date"].dt.strftime("%Y-%m")
    df["month_of_year"] = df["date"].dt.month.astype(str)
    df["time_index"] = ((df["date"].dt.year - df["date"].dt.year.min()) * 12 + df["date"].dt.month).astype(int)
    df["did"] = df["southwest_group"] * df["post_padma"]

    before_after = (
        df.groupby(["southwest_group", "post_padma"])["ntl_mean"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    before_after.to_csv(tables_dir / "ntl_before_after_groups.csv", index=False)

    # District and calendar-month fixed effects; SEs clustered by district.
    did_model = smf.ols("ntl_mean ~ did + C(district) + C(month_id)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["district"]}
    )
    conf = did_model.conf_int().loc["did"]
    did_summary = {
        "did_coefficient": float(did_model.params["did"]),
        "clustered_standard_error": float(did_model.bse["did"]),
        "p_value": float(did_model.pvalues["did"]),
        "ci_95_low": float(conf.iloc[0]),
        "ci_95_high": float(conf.iloc[1]),
    }

    pre = df[df["post_padma"] == 0].copy()
    pretrend = smf.ols(
        "ntl_mean ~ time_index + southwest_group:time_index + C(district) + C(month_of_year)", data=pre
    ).fit(cov_type="cluster", cov_kwds={"groups": pre["district"]})
    did_summary["pretrend_interaction_coefficient"] = float(pretrend.params["southwest_group:time_index"])
    did_summary["pretrend_p_value"] = float(pretrend.pvalues["southwest_group:time_index"])
    did_summary["parallel_trends_warning"] = bool(pretrend.pvalues["southwest_group:time_index"] < 0.05)
    did_summary["interpretation_limit"] = "DiD is credible only if pre-trends are sufficiently similar and no simultaneous group-specific shock drives the estimate."

    (reports_dir / "ntl_did_summary.json").write_text(json.dumps(did_summary, indent=2), encoding="utf-8")
    # With 64 district clusters, statsmodels warns that the huge omnibus F-test
    # over all fixed-effect dummies is rank deficient. This does not invalidate
    # the reported DiD coefficient and its district-clustered standard error.
    # Suppress only that summary-rendering warning; coefficient results remain intact.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank.*")
        did_text = did_model.summary().as_text()
        pretrend_text = pretrend.summary().as_text()
    (reports_dir / "ntl_did_model_summary.txt").write_text(did_text, encoding="utf-8")
    (reports_dir / "ntl_pretrend_model_summary.txt").write_text(pretrend_text, encoding="utf-8")

    trend = df.groupby(["date", "southwest_group"], as_index=False)["ntl_mean"].mean()
    trend["group"] = trend["southwest_group"].map({0: "Control districts", 1: "Southwestern districts"})
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=trend, x="date", y="ntl_mean", hue="group")
    plt.axvline(pd.Timestamp("2022-07-01"), color="black", linestyle="--", label="First full post-opening month")
    plt.ylabel("Mean nighttime-light intensity")
    plt.title("Nighttime-light trends: southwest vs control")
    plt.tight_layout()
    plt.savefig(figures_dir / "ntl_treatment_control_trends.png", dpi=180)
    plt.close()
    return did_summary
