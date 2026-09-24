from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parent
METRIC_DIR = ROOT / "outputs" / "metrics"

SEEDS = [1, 7, 42]


CANDIDATES = [

    {
        "task": "traffic_h1",
        "model": "GRU",
        "target": "Total_Traffic",
        "window": 14,
        "horizon": 1,
        "hidden_size": 96,
        "dropout": 0.10,
    },

    {
        "task": "cash_h1",
        "model": "GRU",
        "target": "Total_Cash",
        "window": 7,
        "horizon": 1,
        "hidden_size": 64,
        "dropout": 0.20,
    },

    {
        "task": "traffic_h7",
        "model": "LSTM",
        "target": "Total_Traffic",
        "window": 7,
        "horizon": 7,
        "hidden_size": 96,
        "dropout": 0.30,
    },

    {
        "task": "cash_h7_lstm",
        "model": "LSTM",
        "target": "Total_Cash",
        "window": 7,
        "horizon": 7,
        "hidden_size": 96,
        "dropout": 0.30,
    },

    {
        "task": "cash_h7_gru",
        "model": "GRU",
        "target": "Total_Cash",
        "window": 60,
        "horizon": 7,
        "hidden_size": 64,
        "dropout": 0.10,
    },
]


rows = []


for candidate in CANDIDATES:

    found = set()

    for file in METRIC_DIR.glob("*.json"):

        try:

            with open(file, "r") as f:
                data = json.load(f)

        except Exception:
            continue


        config = data.get("config", {})
        val = data.get("validation")


        if val is None:
            continue


        match = (

            config.get("input_type") == "raw"

            and
            config.get("model") == candidate["model"]

            and
            config.get("target") == candidate["target"]

            and
            config.get("window") == candidate["window"]

            and
            config.get("horizon") == candidate["horizon"]

            and
            config.get("hidden_size") == candidate["hidden_size"]

            and
            abs(
                float(config.get("dropout", -1))
                -
                candidate["dropout"]
            ) < 1e-9

            and
            config.get("seed") in SEEDS
        )


        if not match:
            continue


        seed = int(config["seed"])

        found.add(seed)


        rows.append({

            "task": candidate["task"],
            "model": candidate["model"],
            "window": candidate["window"],
            "hidden_size": candidate["hidden_size"],
            "dropout": candidate["dropout"],
            "seed": seed,

            "MAE": val["MAE"],
            "RMSE": val["RMSE"],
            "MAPE": val["MAPE"],
            "R2": val["R2"],
        })


    missing = set(SEEDS) - found

    if missing:

        raise RuntimeError(
            f"Missing seeds {sorted(missing)} "
            f"for {candidate['task']}"
        )


details = pd.DataFrame(rows)


summary = (

    details

    .groupby(
        [
            "task",
            "model",
            "window",
            "hidden_size",
            "dropout",
        ],
        as_index=False,
    )

    .agg(

        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),

        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),

        MAPE_mean=("MAPE", "mean"),
        MAPE_std=("MAPE", "std"),

        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
    )
)


summary = summary.sort_values(
    "task"
)


pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)


print(
    "\n"
    + "=" * 120
)

print(
    "FINAL RAW CANDIDATES - MULTI-SEED VALIDATION"
)

print(
    "=" * 120
)


print(
    summary.to_string(
        index=False
    )
)


details_path = (
    METRIC_DIR
    / "raw_final_candidates_detailed.csv"
)

summary_path = (
    METRIC_DIR
    / "raw_final_candidates_summary.csv"
)


details.to_csv(
    details_path,
    index=False
)

summary.to_csv(
    summary_path,
    index=False
)


print("\nSaved:")
print(details_path)
print(summary_path)