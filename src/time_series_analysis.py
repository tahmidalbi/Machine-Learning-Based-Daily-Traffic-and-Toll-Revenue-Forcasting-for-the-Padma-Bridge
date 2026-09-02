from __future__ import annotations

import json
import contextlib
import io
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_padma_jamuna(daily_path: Path, tables_dir: Path, figures_dir: Path, reports_dir: Path):
    try:
        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.stattools import adfuller, grangercausalitytests
    except ImportError:
        warnings.warn("statsmodels unavailable; Padma-Jamuna Granger/VAR analysis skipped.")
        return {"status": "skipped: statsmodels unavailable"}

    raw = pd.read_csv(daily_path, parse_dates=["date"])
    raw = raw.set_index("date")[["padma_total_traffic", "jamuna_total_traffic"]].dropna()
    log_levels = np.log1p(raw)
    # VAR only needs equally spaced observations in row order. Resetting the
    # irregular DatetimeIndex avoids statsmodels treating it as a forecast index.
    stationary = log_levels.diff().dropna().reset_index(drop=True)

    adf_rows = []
    for column in log_levels:
        for transformation, values in [("log_level", log_levels[column]), ("log_first_difference", stationary[column])]:
            stat, pvalue, used_lag, nobs, *_ = adfuller(values.dropna(), autolag="AIC")
            adf_rows.append(
                {"series": column, "transformation": transformation, "ADF_statistic": stat, "p_value": pvalue, "used_lag": used_lag, "n": nobs}
            )
    pd.DataFrame(adf_rows).to_csv(tables_dir / "padma_jamuna_adf_tests.csv", index=False)

    correlation_rows = []
    for lag in range(-14, 15):
        # Positive lag means Jamuna occurs earlier and is compared with later Padma.
        corr = stationary["padma_total_traffic"].corr(stationary["jamuna_total_traffic"].shift(lag))
        correlation_rows.append({"lag_days": lag, "correlation": corr, "meaning": "positive lag = Jamuna leads Padma"})
    correlation = pd.DataFrame(correlation_rows)
    correlation.to_csv(tables_dir / "padma_jamuna_cross_correlation.csv", index=False)
    plt.figure(figsize=(10, 5))
    plt.stem(correlation["lag_days"], correlation["correlation"], basefmt=" ")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Lag in days (positive = Jamuna leads Padma)")
    plt.ylabel("Correlation of log traffic changes")
    plt.title("Padma-Jamuna cross-correlation")
    plt.tight_layout()
    plt.savefig(figures_dir / "padma_jamuna_cross_correlation.png", dpi=180)
    plt.close()

    granger_rows = []
    max_lag = 14
    for target, cause in [("padma_total_traffic", "jamuna_total_traffic"), ("jamuna_total_traffic", "padma_total_traffic")]:
        # Omitting deprecated ``verbose`` works on current statsmodels. Redirecting
        # stdout also keeps compatibility with older releases that printed by default.
        with contextlib.redirect_stdout(io.StringIO()):
            tests = grangercausalitytests(stationary[[target, cause]], maxlag=max_lag)
        for lag, result in tests.items():
            f_stat, p_value, df_denom, df_num = result[0]["ssr_ftest"]
            granger_rows.append(
                {"cause": cause, "target": target, "lag": lag, "F_statistic": f_stat, "p_value": p_value, "df_num": df_num, "df_denom": df_denom}
            )
    granger = pd.DataFrame(granger_rows)
    granger.to_csv(tables_dir / "padma_jamuna_granger.csv", index=False)

    lag_selection = VAR(stationary).select_order(maxlags=14)
    chosen_lag = lag_selection.aic
    if chosen_lag is None or not np.isfinite(chosen_lag):
        chosen_lag = 7
    chosen_lag = max(1, int(chosen_lag))
    var_result = VAR(stationary).fit(chosen_lag)
    # VARSummary implements __str__ but, unlike RegressionResults.summary(),
    # does not implement as_text() in current statsmodels releases.
    (reports_dir / "padma_jamuna_var_summary.txt").write_text(str(var_result.summary()), encoding="utf-8")
    irf = var_result.irf(14)
    figure = irf.plot(orth=False)
    figure.set_size_inches(10, 8)
    figure.tight_layout()
    figure.savefig(figures_dir / "padma_jamuna_impulse_response.png", dpi=180)
    plt.close(figure)

    summary = {
        "transformation": "log1p then first difference, used to reduce spurious regression from shared trends",
        "var_lag_selected_by_aic": chosen_lag,
        "smallest_granger_p_values": granger.groupby(["cause", "target"])["p_value"].min().reset_index().to_dict("records"),
        "interpretation_limit": "Granger causality is predictive temporal dependence, not physical causation.",
    }
    (reports_dir / "padma_jamuna_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
