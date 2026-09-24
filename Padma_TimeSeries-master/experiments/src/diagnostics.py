from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from data_utils import prepare_padma, PROJECT_ROOT


FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def run_stationarity_tests(series, name):
    series = series.dropna()

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # ---------------- ADF ----------------
    adf_result = adfuller(series, autolag="AIC")

    print("\nADF test")
    print("Statistic:", adf_result[0])
    print("p-value:", adf_result[1])

    if adf_result[1] < 0.05:
        print("ADF interpretation: evidence for stationarity.")
    else:
        print("ADF interpretation: cannot reject non-stationarity.")

    # ---------------- KPSS ----------------
    kpss_result = kpss(
        series,
        regression="c",
        nlags="auto"
    )

    print("\nKPSS test")
    print("Statistic:", kpss_result[0])
    print("p-value:", kpss_result[1])

    if kpss_result[1] < 0.05:
        print("KPSS interpretation: evidence of non-stationarity.")
    else:
        print("KPSS interpretation: stationarity is not rejected.")


def create_plots(df, target):
    temp = df[["Date", target]].dropna().copy()
    temp = temp.set_index("Date")

    # ========================================================
    # Original + rolling statistics
    # ========================================================

    rolling_mean = temp[target].rolling(30).mean()
    rolling_std = temp[target].rolling(30).std()

    plt.figure(figsize=(14, 6))
    plt.plot(temp.index, temp[target], label="Original", alpha=0.55)
    plt.plot(
        rolling_mean.index,
        rolling_mean,
        label="30-day rolling mean"
    )
    plt.legend()
    plt.title(f"{target}: original series and rolling mean")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"{target}_rolling_mean.png",
        dpi=160
    )
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(
        rolling_std.index,
        rolling_std,
        label="30-day rolling standard deviation"
    )
    plt.legend()
    plt.title(f"{target}: rolling variability")
    plt.xlabel("Date")
    plt.ylabel("Rolling standard deviation")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"{target}_rolling_std.png",
        dpi=160
    )
    plt.close()

    # ========================================================
    # ACF
    # ========================================================

    plt.figure(figsize=(12, 5))
    plot_acf(
        temp[target],
        lags=60,
        ax=plt.gca()
    )
    plt.title(f"{target}: ACF")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"{target}_acf.png",
        dpi=160
    )
    plt.close()

    # ========================================================
    # PACF
    # ========================================================

    plt.figure(figsize=(12, 5))
    plot_pacf(
        temp[target],
        lags=60,
        method="ywm",
        ax=plt.gca()
    )
    plt.title(f"{target}: PACF")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"{target}_pacf.png",
        dpi=160
    )
    plt.close()

    # ========================================================
    # Weekly seasonal decomposition
    # ========================================================

    # Interpolation here is ONLY for visualization/
    # decomposition, not model target creation.
    daily = temp[target].asfreq("D")
    decomposition_input = daily.interpolate(
        method="time",
        limit_direction="both"
    )

    result = seasonal_decompose(
        decomposition_input,
        model="additive",
        period=7
    )

    fig = result.plot()
    fig.set_size_inches(14, 9)
    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / f"{target}_decomposition.png",
        dpi=160
    )
    plt.close(fig)


def main():
    df = prepare_padma()

    for target in [
        "Total_Traffic",
        "Total_Cash"
    ]:
        run_stationarity_tests(
            df[target],
            target
        )

        create_plots(
            df,
            target
        )


if __name__ == "__main__":
    main()