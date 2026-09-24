from pathlib import Path

RANDOM_SEED = 42
RAIL_COMMERCIAL_START = "2023-11-01"
PADMA_OPENING_MONTH = "2022-07-01"

PADMA_FILE = "padma_toll_report_with_holidays_weather.csv"
JAMUNA_FILE = "jamuna_toll_report.csv"
NTL_FILE = "bangladesh_ntl_2019_2025_padma_groups.csv"

TARGETS = {
    "traffic": "padma_total_traffic",
    "toll": "padma_total_cash",
}

EXTERNAL_FEATURES = [
    "time_index",
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
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "rainfall_mm",
    "humidity_pct",
    "wind_speed_kmh",
    "rainy_day",
    "heavy_rain",
    "rail_open",
    "time_after_rail",
]

HISTORY_FEATURES = [
    "padma_traffic_lag_1",
    "padma_traffic_lag_2",
    "padma_traffic_lag_3",
    "padma_traffic_lag_7",
    "padma_traffic_lag_14",
    "padma_traffic_lag_28",
    "padma_traffic_roll_mean_7",
    "padma_traffic_roll_mean_14",
    "padma_traffic_roll_mean_30",
    "padma_traffic_roll_std_7",
    "padma_traffic_roll_std_30",
    "padma_traffic_change_1_7",
    "padma_roll7_minus_roll30",
    "padma_cash_lag_1",
    "padma_cash_lag_7",
    "padma_cash_lag_14",
    "padma_cash_roll_mean_7",
    "padma_cash_roll_mean_30",
    "revenue_per_vehicle_lag_1",
    "revenue_per_vehicle_roll_mean_7",
    "jamuna_traffic_lag_1",
    "jamuna_traffic_lag_2",
    "jamuna_traffic_lag_7",
    "jamuna_traffic_lag_14",
    "jamuna_traffic_roll_mean_7",
    "jamuna_traffic_roll_mean_30",
    "jamuna_cash_lag_1",
    "jamuna_cash_lag_7",
    "jamuna_cash_roll_mean_7",
]

MODEL_FEATURES = EXTERNAL_FEATURES + HISTORY_FEATURES

SARIMAX_FEATURES = [
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "weekend",
    "is_holiday",
    "eid",
    "days_to_nearest_eid",
    "temp_mean_c",
    "rainfall_mm",
    "humidity_pct",
    "wind_speed_kmh",
    "rainy_day",
    "heavy_rain",
    "rail_open",
    "time_after_rail",
]


def ensure_directories(project_root: Path) -> dict[str, Path]:
    paths = {
        "processed": project_root / "data" / "processed",
        "models": project_root / "models",
        "outputs": project_root / "outputs",
        "figures": project_root / "outputs" / "figures",
        "tables": project_root / "outputs" / "tables",
        "reports": project_root / "outputs" / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

