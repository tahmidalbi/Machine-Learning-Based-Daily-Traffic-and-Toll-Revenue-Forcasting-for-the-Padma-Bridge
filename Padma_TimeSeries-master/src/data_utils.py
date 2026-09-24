from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PADMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "padma_toll_report_with_holidays_weather.csv"
)

JAMUNA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "jamuna_toll_report.csv"
)


# ============================================================
# NUMERIC CLEANING
# ============================================================

def clean_numeric_column(series):
    """
    Convert values such as:
        '51,316'
        '2,09,31,550.00'
    into proper numeric values.

    Invalid/empty values become NaN.
    """

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({
            "": np.nan,
            "nan": np.nan,
            "None": np.nan
        })
    )

    return pd.to_numeric(cleaned, errors="coerce")


# ============================================================
# PADMA DATA
# ============================================================

def load_padma():
    df = pd.read_csv(PADMA_PATH)

    # Padma dates look like 26-06-22
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d-%m-%y",
        errors="raise"
    )

    numeric_text_columns = [
        "Traffic_Mawa",
        "Traffic_Jajira",
        "Cash_Mawa",
        "Cash_Jajira",
        "Total_Traffic",
        "Total_Cash",
    ]

    for col in numeric_text_columns:
        df[col] = clean_numeric_column(df[col])

    df = (
        df.sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# JAMUNA DATA
# ============================================================

def load_jamuna():
    df = pd.read_csv(JAMUNA_PATH)

    # Jamuna dates look like 01/07/2022
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d/%m/%Y",
        errors="raise"
    )

    numeric_columns = [
        "Traffic_East",
        "Traffic_West",
        "Cash_East",
        "Cash_West",
        "Total_Traffic",
        "Total_Cash",
    ]

    for col in numeric_columns:
        df[col] = clean_numeric_column(df[col])

    df = (
        df.sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# DATE / CALENDAR FEATURES
# ============================================================

def add_calendar_features(df):
    df = df.copy()

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["day_of_year"] = df["Date"].dt.dayofyear

    # Cyclic encoding.
    # Monday and Sunday should be considered close to one another.
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )
    df["month_cos"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
    )

    df["doy_sin"] = np.sin(
        2 * np.pi * (df["day_of_year"] - 1) / 365.25
    )
    df["doy_cos"] = np.cos(
        2 * np.pi * (df["day_of_year"] - 1) / 365.25
    )

    # Simple time trend
    df["time_index"] = (
        df["Date"] - df["Date"].min()
    ).dt.days

    return df


# ============================================================
# SIGNED EID RELATIVE DAY
# ============================================================

def add_signed_eid_day(df):
    """
    Existing dataset contains days_to_nearest_eid, but we also
    create signed distance:

        -3 = three days before Eid
         0 = Eid
        +3 = three days after Eid

    Eid anchor dates are found from holiday_name entries that
    correspond to the actual Eid date.
    """

    df = df.copy()

    eid_names = {
        "Eid al-Fitr",
        "Eid-ul-Azha",
    }

    eid_dates = (
        df.loc[
            df["holiday_name"].isin(eid_names),
            "Date"
        ]
        .dropna()
        .sort_values()
        .tolist()
    )

    if len(eid_dates) == 0:
        df["eid_relative_day"] = 0
        return df

    def signed_distance(date):
        differences = [
            (date - eid_date).days
            for eid_date in eid_dates
        ]

        return min(
            differences,
            key=lambda x: abs(x)
        )

    df["eid_relative_day"] = (
        df["Date"]
        .apply(signed_distance)
        .astype(float)
    )

    return df


# ============================================================
# RAILWAY FEATURES
# ============================================================

def add_railway_features(df):
    df = df.copy()

    railway_start = pd.Timestamp("2023-11-01")

    df["railway_open"] = (
        df["Date"] >= railway_start
    ).astype(int)

    df["days_since_railway"] = (
        df["Date"] - railway_start
    ).dt.days.clip(lower=0)

    return df


# ============================================================
# CONTINUOUS CALENDAR
# ============================================================

def add_continuous_calendar(df):
    """
    Makes missing calendar dates explicit.

    IMPORTANT:
    We do NOT invent missing traffic/toll values.
    Missing rows remain NaN and sequence construction later
    skips samples containing unavailable information.
    """

    df = df.copy()

    all_dates = pd.date_range(
        df["Date"].min(),
        df["Date"].max(),
        freq="D"
    )

    df = (
        df.set_index("Date")
        .reindex(all_dates)
        .rename_axis("Date")
        .reset_index()
    )

    return df


# ============================================================
# PREPARE BASIC PADMA DATA
# ============================================================

def prepare_padma():
    df = load_padma()

    df = add_continuous_calendar(df)
    df = add_calendar_features(df)
    df = add_signed_eid_day(df)
    df = add_railway_features(df)

    return df


if __name__ == "__main__":
    padma = prepare_padma()

    print(padma.head())
    print()
    print("Rows:", len(padma))
    print("Date range:")
    print(padma["Date"].min(), "to", padma["Date"].max())

    print("\nMissing target rows:")
    print(
        padma[
            padma["Total_Traffic"].isna()
        ][["Date", "Total_Traffic"]]
    )