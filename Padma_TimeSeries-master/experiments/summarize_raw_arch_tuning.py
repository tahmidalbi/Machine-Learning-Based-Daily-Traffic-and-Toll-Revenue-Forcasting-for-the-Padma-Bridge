from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parent

METRIC_DIR = (
    ROOT
    / "outputs"
    / "metrics"
)


TUNING_SEED = 123


CANDIDATES = [

    {
        "task": "traffic_h1",
        "model": "GRU",
        "target": "Total_Traffic",
        "window": 14,
        "horizon": 1,
    },

    {
        "task": "cash_h1",
        "model": "GRU",
        "target": "Total_Cash",
        "window": 7,
        "horizon": 1,
    },

    {
        "task": "traffic_h7",
        "model": "LSTM",
        "target": "Total_Traffic",
        "window": 7,
        "horizon": 7,
    },

    {
        "task": "cash_h7_gru",
        "model": "GRU",
        "target": "Total_Cash",
        "window": 60,
        "horizon": 7,
    },

    {
        "task": "cash_h7_lstm",
        "model": "LSTM",
        "target": "Total_Cash",
        "window": 7,
        "horizon": 7,
    },
]


rows = []


for file in METRIC_DIR.glob(
    f"*_raw_*_seed{TUNING_SEED}.json"
):

    with open(
        file,
        "r"
    ) as f:

        data = json.load(f)


    config = data.get(
        "config",
        {}
    )

    validation = data.get(
        "validation"
    )


    if validation is None:

        continue


    for candidate in CANDIDATES:

        matches = (

            config.get("input_type")
            == "raw"

            and

            config.get("model")
            == candidate["model"]

            and

            config.get("target")
            == candidate["target"]

            and

            config.get("window")
            == candidate["window"]

            and

            config.get("horizon")
            == candidate["horizon"]

            and

            config.get("seed")
            == TUNING_SEED
        )


        if not matches:

            continue


        rows.append({

            "task":
                candidate[
                    "task"
                ],

            "model":
                config[
                    "model"
                ],

            "window":
                config[
                    "window"
                ],

            "horizon":
                config[
                    "horizon"
                ],

            "hidden_size":
                config[
                    "hidden_size"
                ],

            "dropout":
                config[
                    "dropout"
                ],

            "MAE":
                validation[
                    "MAE"
                ],

            "RMSE":
                validation[
                    "RMSE"
                ],

            "MAPE":
                validation[
                    "MAPE"
                ],

            "R2":
                validation[
                    "R2"
                ],
        })


if not rows:

    raise RuntimeError(
        "No raw architecture tuning results found."
    )


results = pd.DataFrame(
    rows
)


results = results.drop_duplicates(

    subset=[
        "task",
        "hidden_size",
        "dropout",
    ],

    keep="last"
)


results = results.sort_values(

    by=[
        "task",
        "MAE",
    ]
)


# ============================================================
# BEST ARCHITECTURE PER CANDIDATE
# ============================================================

best = (

    results

    .sort_values(
        "MAE"
    )

    .groupby(
        "task",
        as_index=False
    )

    .first()
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
    "STRICT RAW ARCHITECTURE TUNING"
)

print(
    "=" * 120
)


for task, group in results.groupby(
    "task"
):

    print(
        "\n\n"
        + "#" * 100
    )

    print(
        task.upper()
    )

    print(
        "#" * 100
    )


    print(

        group[
            [
                "model",
                "window",
                "hidden_size",
                "dropout",
                "MAE",
                "RMSE",
                "MAPE",
                "R2",
            ]
        ]

        .sort_values(
            "MAE"
        )

        .to_string(
            index=False
        )
    )


print(
    "\n\n"
    + "=" * 120
)

print(
    "BEST ARCHITECTURE PER CANDIDATE"
)

print(
    "=" * 120
)


print(

    best[
        [
            "task",
            "model",
            "window",
            "hidden_size",
            "dropout",
            "MAE",
            "RMSE",
            "MAPE",
            "R2",
        ]
    ]

    .to_string(
        index=False
    )
)


results_path = (

    METRIC_DIR
    / "raw_arch_tuning_summary.csv"
)


best_path = (

    METRIC_DIR
    / "raw_arch_tuning_best.csv"
)


results.to_csv(
    results_path,
    index=False
)


best.to_csv(
    best_path,
    index=False
)


print(
    "\nSaved:"
)

print(
    results_path
)

print(
    best_path
)