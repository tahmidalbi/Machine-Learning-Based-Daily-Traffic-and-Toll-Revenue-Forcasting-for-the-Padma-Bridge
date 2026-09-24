from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    JAMUNA_FILE,
    MODEL_FEATURES,
    NTL_FILE,
    PADMA_FILE,
    PADMA_OPENING_MONTH,
    RAIL_COMMERCIAL_START,
    TARGETS,
)


PADMA_RENAME = {
    "Date": "date",
    "Traffic_Mawa": "padma_mawa_traffic",
    "Traffic_Jajira": "padma_jajira_traffic",
    "Cash_Mawa": "padma_mawa_cash",
    "Cash_Jajira": "padma_jajira_cash",
    "Total_Traffic": "padma_total_traffic",
    "Total_Cash": "padma_total_cash",
}

JAMUNA_RENAME = {
    "Date": "date",
    "Traffic_East": "jamuna_east_traffic",
    "Traffic_West": "jamuna_west_traffic",
    "Cash_East": "jamuna_east_cash",
    "Cash_West": "jamuna_west_cash",
    "Total_Traffic": "jamuna_total_traffic",
    "Total_Cash": "jamuna_total_cash",
}


def resolve_input(data_dir: Path, canonical_name: str) -> Path:
    direct = data_dir / canonical_name
    if direct.exists():
        return direct
    matches = sorted(data_dir.glob(f"*{canonical_name}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Missing {canonical_name} in {data_dir.resolve()}")
    raise RuntimeError(f"More than one file ends with {canonical_name}: {matches}")


def parse_integer(value) -> float:
    if pd.isna(value):
        return np.nan
    digits = re.sub(r"\D", "", str(value).strip())
    return float(digits) if digits else np.nan


def parse_money(value) -> float:
    """Parse Indian grouping and malformed separators while preserving final cents."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if "." in text and re.fullmatch(r"\d{2}", text.rsplit(".", 1)[1]):
        integer_part, cents = text.rsplit(".", 1)
        digits = re.sub(r"\D", "", integer_part)
        return float(f"{digits}.{cents}") if digits else np.nan
    digits = re.sub(r"\D", "", text)
    return float(digits) if digits else np.nan


def _bridge_audit(
    df: pd.DataFrame,
    label: str,
    side_traffic: tuple[str, str],
    total_traffic: str,
    side_cash: tuple[str, str],
    total_cash: str,
) -> dict:
    traffic_diff = df[side_traffic[0]] + df[side_traffic[1]] - df[total_traffic]
    cash_diff = df[side_cash[0]] + df[side_cash[1]] - df[total_cash]
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing_dates = full_dates.difference(df["date"])
    return {
        "dataset": label,
        "rows": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "duplicate_dates": int(df["date"].duplicated().sum()),
        "missing_dates": [str(x.date()) for x in missing_dates],
        "traffic_total_mismatch_count": int((traffic_diff.abs() > 0).sum()),
        "traffic_total_mismatch_dates": [str(x.date()) for x in df.loc[traffic_diff.abs() > 0, "date"]],
        "cash_total_mismatch_count": int((cash_diff.abs() > 0.01).sum()),
        "cash_total_mismatch_dates": [str(x.date()) for x in df.loc[cash_diff.abs() > 0.01, "date"]],
        "policy": "Totals are preserved as reported; inconsistencies are flagged, never silently rewritten.",
    }


def load_padma(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path).rename(columns=PADMA_RENAME)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%y", errors="raise")
    traffic_cols = ["padma_mawa_traffic", "padma_jajira_traffic", "padma_total_traffic"]
    money_cols = ["padma_mawa_cash", "padma_jajira_cash", "padma_total_cash"]
    for col in traffic_cols:
        df[col] = df[col].map(parse_integer)
    for col in money_cols:
        df[col] = df[col].map(parse_money)
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    audit = _bridge_audit(
        df,
        "Padma",
        ("padma_mawa_traffic", "padma_jajira_traffic"),
        "padma_total_traffic",
        ("padma_mawa_cash", "padma_jajira_cash"),
        "padma_total_cash",
    )
    return df, audit


def load_jamuna(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path).rename(columns=JAMUNA_RENAME)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="raise")
    traffic_cols = ["jamuna_east_traffic", "jamuna_west_traffic", "jamuna_total_traffic"]
    money_cols = ["jamuna_east_cash", "jamuna_west_cash", "jamuna_total_cash"]
    for col in traffic_cols:
        df[col] = df[col].map(parse_integer)
    for col in money_cols:
        df[col] = df[col].map(parse_money)
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    audit = _bridge_audit(
        df,
        "Jamuna",
        ("jamuna_east_traffic", "jamuna_west_traffic"),
        "jamuna_total_traffic",
        ("jamuna_east_cash", "jamuna_west_cash"),
        "jamuna_total_cash",
    )
    return df, audit


def load_ntl(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    required = {"NAM_1", "NAM_2", "date", "ntl_mean", "southwest_group", "post_padma"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"NTL file is missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    expected_post = (df["date"] >= pd.Timestamp(PADMA_OPENING_MONTH)).astype(int)
    audit = {
        "dataset": "Nighttime lights",
        "rows": int(len(df)),
        "districts": int(df["NAM_2"].nunique()),
        "months": int(df["date"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "duplicate_district_months": int(df.duplicated(["NAM_2", "date"]).sum()),
        "post_padma_disagreements": int((df["post_padma"] != expected_post).sum()),
        "southwest_districts": sorted(df.loc[df["southwest_group"] == 1, "NAM_2"].unique().tolist()),
    }
    return df.sort_values(["date", "NAM_2"]).reset_index(drop=True), audit


def _nearest_eid_relative_day(dates: pd.Series, eid_dates: pd.DatetimeIndex) -> pd.Series:
    if len(eid_dates) == 0:
        return pd.Series(np.nan, index=dates.index)
    eid_values = eid_dates.values.astype("datetime64[D]")
    date_values = dates.values.astype("datetime64[D]")
    output = []
    for value in date_values:
        deltas = (value - eid_values).astype("timedelta64[D]").astype(int)
        output.append(int(deltas[np.argmin(np.abs(deltas))]))
    return pd.Series(output, index=dates.index, dtype=float)


def add_features(daily: pd.DataFrame, eid_dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    df = daily.sort_values("date").copy()
    if eid_dates is None:
        eid_dates = pd.DatetimeIndex(
            df.loc[df.get("days_to_nearest_eid", pd.Series(index=df.index, dtype=float)).eq(0), "date"].dropna().unique()
        )

    df["time_index"] = (df["date"] - df["date"].min()).dt.days.astype(float)
    dow = df["date"].dt.dayofweek
    month = df["date"].dt.month
    doy = df["date"].dt.dayofyear
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    df["doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.25)
    df["weekend"] = df.get("weekend", dow.isin([4, 5]).astype(int)).fillna(dow.isin([4, 5]).astype(int))
    for binary in ["is_holiday", "eid"]:
        if binary not in df:
            df[binary] = 0
        df[binary] = df[binary].fillna(0).astype(int)
    if "days_to_nearest_eid" not in df:
        df["days_to_nearest_eid"] = np.nan
    calculated_distance = _nearest_eid_relative_day(df["date"], eid_dates)
    df["days_to_nearest_eid"] = df["days_to_nearest_eid"].fillna(calculated_distance.abs())
    df["eid_relative_day"] = calculated_distance

    df["rainy_day"] = (df["rainfall_mm"] > 0).astype(int)
    df["heavy_rain"] = (df["rainfall_mm"] >= 20).astype(int)
    rail_date = pd.Timestamp(RAIL_COMMERCIAL_START)
    df["rail_open"] = (df["date"] >= rail_date).astype(int)
    df["time_after_rail"] = (df["date"] - rail_date).dt.days.clip(lower=0).astype(float)

    traffic = df["padma_total_traffic"]
    cash = df["padma_total_cash"]
    jamuna_traffic = df["jamuna_total_traffic"]
    jamuna_cash = df["jamuna_total_cash"]
    revenue_per_vehicle = cash / traffic.replace(0, np.nan)

    for lag in [1, 2, 3, 7, 14, 28]:
        df[f"padma_traffic_lag_{lag}"] = traffic.shift(lag)
    for window in [7, 14, 30]:
        shifted = traffic.shift(1)
        df[f"padma_traffic_roll_mean_{window}"] = shifted.rolling(window, min_periods=window).mean()
    for window in [7, 30]:
        df[f"padma_traffic_roll_std_{window}"] = traffic.shift(1).rolling(window, min_periods=window).std()
    df["padma_traffic_change_1_7"] = traffic.shift(1) - traffic.shift(7)
    df["padma_roll7_minus_roll30"] = df["padma_traffic_roll_mean_7"] - df["padma_traffic_roll_mean_30"]

    for lag in [1, 7, 14]:
        df[f"padma_cash_lag_{lag}"] = cash.shift(lag)
    for window in [7, 30]:
        df[f"padma_cash_roll_mean_{window}"] = cash.shift(1).rolling(window, min_periods=window).mean()
    df["revenue_per_vehicle_lag_1"] = revenue_per_vehicle.shift(1)
    df["revenue_per_vehicle_roll_mean_7"] = revenue_per_vehicle.shift(1).rolling(7, min_periods=7).mean()

    for lag in [1, 2, 7, 14]:
        df[f"jamuna_traffic_lag_{lag}"] = jamuna_traffic.shift(lag)
    for window in [7, 30]:
        df[f"jamuna_traffic_roll_mean_{window}"] = jamuna_traffic.shift(1).rolling(window, min_periods=window).mean()
    for lag in [1, 7]:
        df[f"jamuna_cash_lag_{lag}"] = jamuna_cash.shift(lag)
    df["jamuna_cash_roll_mean_7"] = jamuna_cash.shift(1).rolling(7, min_periods=7).mean()
    return df


def build_datasets(data_dir: Path, processed_dir: Path, reports_dir: Path) -> dict:
    padma_path = resolve_input(data_dir, PADMA_FILE)
    jamuna_path = resolve_input(data_dir, JAMUNA_FILE)
    ntl_path = resolve_input(data_dir, NTL_FILE)

    padma, padma_audit = load_padma(padma_path)
    jamuna, jamuna_audit = load_jamuna(jamuna_path)
    ntl, ntl_audit = load_ntl(ntl_path)
    eid_dates = pd.DatetimeIndex(padma.loc[padma["days_to_nearest_eid"] == 0, "date"].unique())

    complete_dates = pd.DataFrame({"date": pd.date_range(padma["date"].min(), padma["date"].max(), freq="D")})
    daily = complete_dates.merge(padma, on="date", how="left", validate="one_to_one")
    daily = daily.merge(jamuna, on="date", how="left", validate="one_to_one")
    daily = add_features(daily, eid_dates=eid_dates)

    required = ["date", *MODEL_FEATURES, *TARGETS.values(), "holiday_name"]
    model_table = daily[[c for c in required if c in daily.columns]].copy()
    complete_model_rows = model_table.dropna(subset=[*MODEL_FEATURES, *TARGETS.values()]).copy()

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(processed_dir / "merged_daily_with_features.csv", index=False)
    complete_model_rows.to_csv(processed_dir / "forecast_modeling_table.csv", index=False)
    ntl.to_csv(processed_dir / "ntl_panel.csv", index=False)

    audit = {
        "selected_source_files": [padma_path.name, jamuna_path.name, ntl_path.name],
        "why_only_three": "These are the richest non-duplicative inputs. Other supplied CSVs are subsets or intermediate duplicates.",
        "padma": padma_audit,
        "jamuna": jamuna_audit,
        "nighttime_lights": ntl_audit,
        "daily_calendar_rows_after_reindex": int(len(daily)),
        "complete_forecasting_rows": int(len(complete_model_rows)),
        "feature_count": int(len(MODEL_FEATURES)),
        "leakage_rule": "Every traffic/toll/Jamuna rolling or lag feature is shifted by at least one day.",
    }
    (reports_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit

