import numpy as np
import pandas as pd

from data_utils import (
    prepare_padma,
    load_jamuna,
)


# ============================================================
# CAUSAL FILL
# ============================================================

def causal_fill(series, seasonal_lag=7):
    """
    Fill missing historical input WITHOUT using future values.

    First:
        try value from same weekday one week earlier.

    Then:
        forward-fill using the most recent past observation.

    No backward filling is used.
    """

    result = series.copy()

    missing = result.isna()

    seasonal_value = result.shift(seasonal_lag)

    result.loc[missing] = seasonal_value.loc[missing]

    # Still missing? Use most recent past value.
    result = result.ffill()

    return result


# ============================================================
# PREPARE JAMUNA ON A CONTINUOUS DAILY CALENDAR
# ============================================================

def prepare_jamuna_for_merge(start_date, end_date):

    jamuna = load_jamuna().copy()

    # Rename so Jamuna and Padma column names never collide.
    jamuna = jamuna.rename(
        columns={
            "Total_Traffic": "Jamuna_Total_Traffic",
            "Total_Cash": "Jamuna_Total_Cash",
            "Traffic_East": "Jamuna_Traffic_East",
            "Traffic_West": "Jamuna_Traffic_West",
            "Cash_East": "Jamuna_Cash_East",
            "Cash_West": "Jamuna_Cash_West",
        }
    )

    # --------------------------------------------------------
    # Create continuous date axis matching Padma period
    # --------------------------------------------------------

    full_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    jamuna = (
        jamuna
        .set_index("Date")
        .reindex(full_dates)
        .rename_axis("Date")
        .reset_index()
    )

    # Indicator:
    # 1 means the original Jamuna observation was unavailable.
    jamuna["jamuna_missing"] = (
        jamuna["Jamuna_Total_Traffic"]
        .isna()
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Keep original values untouched, but make causal
    # historical-input versions.
    # --------------------------------------------------------

    jamuna["Jamuna_Total_Traffic_input"] = causal_fill(
        jamuna["Jamuna_Total_Traffic"]
    )

    jamuna["Jamuna_Total_Cash_input"] = causal_fill(
        jamuna["Jamuna_Total_Cash"]
    )

    return jamuna[
        [
            "Date",
            "Jamuna_Total_Traffic",
            "Jamuna_Total_Cash",
            "Jamuna_Total_Traffic_input",
            "Jamuna_Total_Cash_input",
            "jamuna_missing",
        ]
    ]


# ============================================================
# ADD CAUSAL PADMA INPUT COLUMNS
# ============================================================

def add_padma_input_columns(df):

    df = df.copy()

    # --------------------------------------------------------
    # Keep original Total_Traffic / Total_Cash untouched.
    #
    # These original columns remain the true prediction target.
    # --------------------------------------------------------

    df["padma_missing"] = (
        df["Total_Traffic"]
        .isna()
        .astype(np.float32)
    )

    df["Total_Traffic_input"] = causal_fill(
        df["Total_Traffic"]
    )

    df["Total_Cash_input"] = causal_fill(
        df["Total_Cash"]
    )

    # --------------------------------------------------------
    # Weather values are historical observations inside the
    # sequence.
    #
    # One missing date should not destroy 30 future sequences.
    # Use past-only filling.
    # --------------------------------------------------------

    weather_columns = [
        "temp_mean_c",
        "temp_max_c",
        "temp_min_c",
        "rainfall_mm",
        "humidity_pct",
        "wind_speed_kmh",
    ]

    df["weather_missing"] = (
        df[weather_columns]
        .isna()
        .any(axis=1)
        .astype(np.float32)
    )

    for col in weather_columns:

        input_col = f"{col}_input"

        df[input_col] = causal_fill(
            df[col]
        )

    # --------------------------------------------------------
    # Historical calendar/event inputs.
    #
    # These are already known calendar/event values, but
    # reindexing may create NaNs on a missing source date.
    # --------------------------------------------------------

    event_columns = [
        "weekend",
        "is_holiday",
        "eid",
        "days_to_nearest_eid",
    ]

    for col in event_columns:

        input_col = f"{col}_input"

        df[input_col] = causal_fill(
            df[col]
        )

    return df


# ============================================================
# PADMA TRAFFIC HISTORY FEATURES
# ============================================================

def add_padma_traffic_features(df):

    df = df.copy()

    traffic = df["Total_Traffic_input"]

    # Explicit lag variables.
    for lag in [1, 2, 3, 7, 14, 28]:

        df[f"traffic_lag_{lag}"] = (
            traffic.shift(lag)
        )

    # Rolling statistics must only use values through t-1.
    traffic_history = traffic.shift(1)

    df["traffic_roll7_mean"] = (
        traffic_history
        .rolling(
            window=7,
            min_periods=7
        )
        .mean()
    )

    df["traffic_roll14_mean"] = (
        traffic_history
        .rolling(
            window=14,
            min_periods=14
        )
        .mean()
    )

    df["traffic_roll30_mean"] = (
        traffic_history
        .rolling(
            window=30,
            min_periods=30
        )
        .mean()
    )

    df["traffic_roll7_std"] = (
        traffic_history
        .rolling(
            window=7,
            min_periods=7
        )
        .std()
    )

    df["traffic_roll30_std"] = (
        traffic_history
        .rolling(
            window=30,
            min_periods=30
        )
        .std()
    )

    # Short-term trend versus longer-term trend.
    df["traffic_roll7_minus_roll30"] = (
        df["traffic_roll7_mean"]
        -
        df["traffic_roll30_mean"]
    )

    # Recent change:
    # yesterday compared with one week ago.
    df["traffic_change_1_7"] = (
        df["traffic_lag_1"]
        -
        df["traffic_lag_7"]
    )

    return df


# ============================================================
# PADMA TOLL HISTORY FEATURES
# ============================================================

def add_padma_cash_features(df):

    df = df.copy()

    cash = df["Total_Cash_input"]

    for lag in [1, 7, 14]:

        df[f"cash_lag_{lag}"] = (
            cash.shift(lag)
        )

    cash_history = cash.shift(1)

    df["cash_roll7_mean"] = (
        cash_history
        .rolling(
            window=7,
            min_periods=7
        )
        .mean()
    )

    df["cash_roll30_mean"] = (
        cash_history
        .rolling(
            window=30,
            min_periods=30
        )
        .mean()
    )

    # --------------------------------------------------------
    # Revenue per vehicle
    # --------------------------------------------------------

    revenue_per_vehicle = (
        df["Total_Cash_input"]
        /
        df["Total_Traffic_input"]
        .replace(0, np.nan)
    )

    df["revenue_per_vehicle_lag1"] = (
        revenue_per_vehicle.shift(1)
    )

    df["revenue_per_vehicle_roll7"] = (
        revenue_per_vehicle
        .shift(1)
        .rolling(
            window=7,
            min_periods=7
        )
        .mean()
    )

    return df


# ============================================================
# JAMUNA HISTORY FEATURES
# ============================================================

def add_jamuna_features(df):

    df = df.copy()

    traffic = df[
        "Jamuna_Total_Traffic_input"
    ]

    cash = df[
        "Jamuna_Total_Cash_input"
    ]

    # Traffic lags
    for lag in [1, 2, 7, 14]:

        df[f"jamuna_traffic_lag_{lag}"] = (
            traffic.shift(lag)
        )

    traffic_history = traffic.shift(1)

    df["jamuna_traffic_roll7"] = (
        traffic_history
        .rolling(
            window=7,
            min_periods=7
        )
        .mean()
    )

    df["jamuna_traffic_roll30"] = (
        traffic_history
        .rolling(
            window=30,
            min_periods=30
        )
        .mean()
    )

    # Jamuna toll history
    for lag in [1, 7]:

        df[f"jamuna_cash_lag_{lag}"] = (
            cash.shift(lag)
        )

    df["jamuna_cash_roll7"] = (
        cash
        .shift(1)
        .rolling(
            window=7,
            min_periods=7
        )
        .mean()
    )

    return df


# ============================================================
# COMPLETE DAILY TIME-SERIES TABLE
# ============================================================

def prepare_timeseries_dataframe():

    # --------------------------------------------------------
    # Padma already has a continuous calendar and its
    # calendar/railway features through prepare_padma().
    # --------------------------------------------------------

    padma = prepare_padma().copy()

    padma = add_padma_input_columns(
        padma
    )

    # --------------------------------------------------------
    # Build Jamuna on exactly the same date axis.
    # --------------------------------------------------------

    jamuna = prepare_jamuna_for_merge(
        start_date=padma["Date"].min(),
        end_date=padma["Date"].max()
    )

    # Date-based merge ONLY.
    df = padma.merge(
        jamuna,
        on="Date",
        how="left"
    )

    # --------------------------------------------------------
    # Add leakage-safe historical features.
    # --------------------------------------------------------

    df = add_padma_traffic_features(
        df
    )

    df = add_padma_cash_features(
        df
    )

    df = add_jamuna_features(
        df
    )

    # Ensure chronological order.
    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# DEBUG
# ============================================================

if __name__ == "__main__":

    df = prepare_timeseries_dataframe()

    print(df.head(40))

    print("\nRows:", len(df))

    print(
        "\nDate range:",
        df["Date"].min(),
        "to",
        df["Date"].max()
    )

    print("\nMissing original Padma targets:")

    print(
        df.loc[
            df["Total_Traffic"].isna(),
            [
                "Date",
                "Total_Traffic",
                "Total_Traffic_input",
                "padma_missing",
            ]
        ]
    )

    print("\nMissing Jamuna source rows:")

    print(
        df.loc[
            df["jamuna_missing"] == 1,
            [
                "Date",
                "Jamuna_Total_Traffic",
                "Jamuna_Total_Traffic_input",
            ]
        ].head(20)
    )