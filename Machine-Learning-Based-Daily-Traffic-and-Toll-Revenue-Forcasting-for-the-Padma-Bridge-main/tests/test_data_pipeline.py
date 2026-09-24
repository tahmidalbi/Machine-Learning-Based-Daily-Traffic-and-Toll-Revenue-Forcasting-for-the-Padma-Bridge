from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES
from src.data_pipeline import parse_integer, parse_money


def test_number_parsers():
    assert parse_integer("6.337") == 6337
    assert parse_integer("51,316") == 51316
    assert parse_money("1,08,87,150.00") == 10887150
    assert parse_money("90.65,950.00") == 9065950


def test_generated_features_are_lagged():
    path = Path(__file__).resolve().parents[1] / "data" / "processed" / "merged_daily_with_features.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    assert np.allclose(df["padma_traffic_lag_1"].iloc[1:], df["padma_total_traffic"].iloc[:-1], equal_nan=True)
    manual = df["padma_total_traffic"].shift(1).rolling(7, min_periods=7).mean()
    assert np.allclose(df["padma_traffic_roll_mean_7"], manual, equal_nan=True)
    assert "padma_total_traffic" not in MODEL_FEATURES
    assert "padma_total_cash" not in MODEL_FEATURES
    assert "padma_mawa_traffic" not in MODEL_FEATURES
    assert "padma_jajira_traffic" not in MODEL_FEATURES


if __name__ == "__main__":
    test_number_parsers()
    test_generated_features_are_lagged()
    print("All data-pipeline tests passed.")

