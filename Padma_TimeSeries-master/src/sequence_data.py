import numpy as np
import pandas as pd


# ============================================================
# RAW SEQUENTIAL FEATURES
# ============================================================
#
# No explicit lag / rolling features.
#
# This still represents ordered multivariate time-series data.
# ============================================================

RAW_SEQUENCE_FEATURES = [

    # Actual historical Padma observations
    "Total_Traffic_input",
    "Total_Cash_input",

    # Actual historical weather observations
    "temp_mean_c_input",
    "temp_max_c_input",
    "temp_min_c_input",
    "rainfall_mm_input",
    "humidity_pct_input",
    "wind_speed_kmh_input",
]
# ============================================================
# FEATURE-ENHANCED SEQUENTIAL FEATURES
# ============================================================
#
# IMPORTANT:
#
# These are still supplied as:
#
#    day 1
#    day 2
#    ...
#    day w
#
# We NEVER flatten the time dimension.
# ============================================================

ENHANCED_SEQUENCE_FEATURES = [

    # ========================================================
    # MAIN HISTORICAL SERIES
    # ========================================================

    "Total_Traffic_input",
    "Total_Cash_input",

    # Jamuna synchronized historical channels
    "Jamuna_Total_Traffic_input",
    "Jamuna_Total_Cash_input",

    # ========================================================
    # HISTORICAL WEATHER
    # ========================================================

    "temp_mean_c_input",
    "temp_max_c_input",
    "temp_min_c_input",
    "rainfall_mm_input",
    "humidity_pct_input",
    "wind_speed_kmh_input",

    # ========================================================
    # CALENDAR / EVENT STATE AT EACH HISTORICAL TIME STEP
    # ========================================================

    "dow_sin",
    "dow_cos",

    "month_sin",
    "month_cos",

    "doy_sin",
    "doy_cos",

    "weekend_input",
    "is_holiday_input",
    "eid_input",
    "days_to_nearest_eid_input",
    "eid_relative_day",

    # Railway
    "railway_open",
    "days_since_railway",

    # ========================================================
    # PADMA TRAFFIC HISTORY
    # ========================================================

    "traffic_lag_1",
    "traffic_lag_2",
    "traffic_lag_3",
    "traffic_lag_7",
    "traffic_lag_14",
    "traffic_lag_28",

    "traffic_roll7_mean",
    "traffic_roll14_mean",
    "traffic_roll30_mean",

    "traffic_roll7_std",
    "traffic_roll30_std",

    "traffic_roll7_minus_roll30",
    "traffic_change_1_7",

    # ========================================================
    # PADMA TOLL HISTORY
    # ========================================================

    "cash_lag_1",
    "cash_lag_7",
    "cash_lag_14",

    "cash_roll7_mean",
    "cash_roll30_mean",

    "revenue_per_vehicle_lag1",
    "revenue_per_vehicle_roll7",

    # ========================================================
    # JAMUNA HISTORY
    # ========================================================

    "jamuna_traffic_lag_1",
    "jamuna_traffic_lag_2",
    "jamuna_traffic_lag_7",
    "jamuna_traffic_lag_14",

    "jamuna_traffic_roll7",
    "jamuna_traffic_roll30",

    "jamuna_cash_lag_1",
    "jamuna_cash_lag_7",
    "jamuna_cash_roll7",

    # ========================================================
    # MISSINGNESS
    # ========================================================

    "padma_missing",
    "jamuna_missing",
    "weather_missing",
]


# ============================================================
# TARGET-DATE INFORMATION KNOWN IN ADVANCE
# ============================================================
#
# These do NOT contain traffic/toll/weather observations.
#
# They are calendar/event facts that are legitimately known
# for a future target date.
# ============================================================

FUTURE_KNOWN_FEATURES = [

    "dow_sin",
    "dow_cos",

    "month_sin",
    "month_cos",

    "doy_sin",
    "doy_cos",

    "weekend",
    "is_holiday",
    "eid",
    "days_to_nearest_eid",
    "eid_relative_day",

    "railway_open",
    "days_since_railway",
]


# ============================================================
# SELECT FEATURES
# ============================================================

def get_sequence_features(input_type):

    input_type = input_type.lower()

    if input_type == "raw":
        return RAW_SEQUENCE_FEATURES

    if input_type == "enhanced":
        return ENHANCED_SEQUENCE_FEATURES

    raise ValueError(
        "input_type must be 'raw' or 'enhanced'"
    )


# ============================================================
# CREATE ORDERED TIME-SERIES WINDOWS
# ============================================================

def create_sequences(
    df,
    target="Total_Traffic",
    window=30,
    horizon=1,
    input_type="enhanced",
    use_future_known=True
):

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    feature_columns = get_sequence_features(
        input_type
    )

    # Check all requested features exist.
    missing_columns = [
        col
        for col in feature_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing sequence columns:\n"
            + "\n".join(missing_columns)
        )

    X_seq = []
    X_future = []
    y = []

    origin_dates = []
    target_dates = []

    for end_idx in range(
        window - 1,
        len(df) - horizon
    ):

        start_idx = (
            end_idx
            - window
            + 1
        )

        target_idx = (
            end_idx
            + horizon
        )

        # ====================================================
        # VERIFY EXACT DAILY TIME ORDER
        # ====================================================

        sequence_dates = df.loc[
            start_idx:end_idx,
            "Date"
        ]

        expected_dates = pd.date_range(
            start=sequence_dates.iloc[0],
            end=sequence_dates.iloc[-1],
            freq="D"
        )

        if len(expected_dates) != window:
            continue

        if not np.array_equal(
            sequence_dates.to_numpy(
                dtype="datetime64[ns]"
            ),
            expected_dates.to_numpy(
                dtype="datetime64[ns]"
            )
        ):
            continue

        # ====================================================
        # CREATE THE TIME-SERIES MATRIX
        #
        # Shape:
        #     window x features
        #
        # Example:
        #     30 x 48
        # ====================================================

        sequence = df.loc[
            start_idx:end_idx,
            feature_columns
        ].to_numpy(
            dtype=np.float32
        )

        # Sequence cannot contain unavailable engineered input.
        if np.isnan(sequence).any():
            continue

        # ====================================================
        # TRUE FUTURE TARGET
        #
        # IMPORTANT:
        # We use the untouched original target column.
        # ====================================================

        target_value = df.loc[
            target_idx,
            target
        ]

        # Missing true target?
        # Never invent ground truth.
        if pd.isna(target_value):
            continue

        origin_date = df.loc[
            end_idx,
            "Date"
        ]

        target_date = df.loc[
            target_idx,
            "Date"
        ]

        # ====================================================
        # VERIFY HORIZON
        # ====================================================

        actual_gap = (
            target_date
            -
            origin_date
        ).days

        if actual_gap != horizon:
            continue

        # ====================================================
        # FUTURE-KNOWN CALENDAR / EVENT VARIABLES
        # ====================================================

        if use_future_known:

            future_values = df.loc[
                target_idx,
                FUTURE_KNOWN_FEATURES
            ].to_numpy(
                dtype=np.float32
            )

            if np.isnan(
                future_values
            ).any():
                continue

            X_future.append(
                future_values
            )

        X_seq.append(
            sequence
        )

        y.append(
            float(target_value)
        )

        origin_dates.append(
            origin_date
        )

        target_dates.append(
            target_date
        )

    X_seq = np.asarray(
        X_seq,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    ).reshape(
        -1,
        1
    )

    if use_future_known:

        X_future = np.asarray(
            X_future,
            dtype=np.float32
        )

    else:

        X_future = np.zeros(
            (
                len(X_seq),
                0
            ),
            dtype=np.float32
        )

    metadata = pd.DataFrame(
        {
            "origin_date":
                origin_dates,

            "target_date":
                target_dates,
        }
    )

    return (
        X_seq,
        X_future,
        y,
        metadata,
        feature_columns
    )