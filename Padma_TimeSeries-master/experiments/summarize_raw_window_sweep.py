from pathlib import Path
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

METRIC_DIR = (
    ROOT
    / "outputs"
    / "metrics"
)


# ============================================================
# ONLY READ OUR WINDOW-SWEEP FILES
# ============================================================

SWEEP_SEED = 123


rows = []


for json_file in METRIC_DIR.glob(
    f"*_raw_*_seed{SWEEP_SEED}.json"
):

    with open(
        json_file,
        "r"
    ) as f:

        data = json.load(
            f
        )


    if (
        "config" not in data
        or
        "validation" not in data
    ):

        continue


    config = data[
        "config"
    ]

    validation = data[
        "validation"
    ]


    # --------------------------------------------------------
    # Make sure this really belongs to the raw sweep.
    # --------------------------------------------------------

    if (
        config.get(
            "input_type"
        )
        !=
        "raw"
    ):

        continue


    if (
        config.get(
            "seed"
        )
        !=
        SWEEP_SEED
    ):

        continue


    rows.append({

        "target":
            config[
                "target"
            ],

        "horizon":
            config[
                "horizon"
            ],

        "model":
            config[
                "model"
            ],

        "window":
            config[
                "window"
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

        "file":
            json_file.name,
    })


# ============================================================
# CHECK
# ============================================================

if len(rows) == 0:

    raise RuntimeError(
        "No raw window-sweep metric files were found."
    )


results = pd.DataFrame(
    rows
)


# ============================================================
# SORT
# ============================================================

results = results.sort_values(

    by=[
        "target",
        "horizon",
        "model",
        "MAE",
    ],

    ascending=[
        True,
        True,
        True,
        True,
    ]
)


# ============================================================
# DISPLAY SETTINGS
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)


# ============================================================
# PRINT FULL TABLE
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "STRICT RAW TIME-SERIES WINDOW SWEEP"
)

print(
    "=" * 120
)


for (
    target,
    horizon
), group in results.groupby(
    [
        "target",
        "horizon"
    ]
):

    print(
        "\n\n"
        + "#" * 100
    )

    print(
        f"{target} | Horizon = {horizon}"
    )

    print(
        "#" * 100
    )


    print(

        group[
            [
                "model",
                "window",
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


# ============================================================
# BEST WINDOW FOR EACH MODEL/TASK
# ============================================================

best_rows = (

    results

    .sort_values(
        "MAE"
    )

    .groupby(
        [
            "target",
            "horizon",
            "model",
        ],

        as_index=False
    )

    .first()
)


print(
    "\n\n"
    + "=" * 120
)

print(
    "BEST WINDOW PER MODEL"
)

print(
    "=" * 120
)


print(

    best_rows[
        [
            "target",
            "horizon",
            "model",
            "window",
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


# ============================================================
# SAVE FULL RESULTS
# ============================================================

full_output_path = (

    METRIC_DIR
    / "raw_window_sweep_summary.csv"
)


results.to_csv(

    full_output_path,

    index=False
)


# ============================================================
# SAVE BEST WINDOWS
# ============================================================

best_output_path = (

    METRIC_DIR
    / "raw_window_sweep_best.csv"
)


best_rows.to_csv(

    best_output_path,

    index=False
)


print(
    "\n\nSaved full results:"
)

print(
    full_output_path
)


print(
    "\nSaved best-window table:"
)

print(
    best_output_path
)