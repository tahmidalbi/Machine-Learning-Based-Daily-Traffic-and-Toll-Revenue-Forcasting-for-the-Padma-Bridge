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
# MULTI-SEED SETTINGS
# ============================================================

SEEDS = [
    1,
    7,
    42,
]


# ============================================================
# CANDIDATES SELECTED FROM RAW WINDOW SWEEP
# ============================================================

CANDIDATES = [

    # --------------------------------------------------------
    # TRAFFIC H1
    # --------------------------------------------------------

    {
        "task": "traffic_h1",
        "target": "Total_Traffic",
        "horizon": 1,
        "model": "GRU",
        "window": 14,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.20,
    },

    {
        "task": "traffic_h1",
        "target": "Total_Traffic",
        "horizon": 1,
        "model": "LSTM",
        "window": 7,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.20,
    },


    # --------------------------------------------------------
    # CASH H1
    # --------------------------------------------------------

    {
        "task": "cash_h1",
        "target": "Total_Cash",
        "horizon": 1,
        "model": "GRU",
        "window": 7,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.20,
    },

    {
        "task": "cash_h1",
        "target": "Total_Cash",
        "horizon": 1,
        "model": "GRU",
        "window": 14,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.20,
    },


    # --------------------------------------------------------
    # TRAFFIC H7
    # --------------------------------------------------------

    {
        "task": "traffic_h7",
        "target": "Total_Traffic",
        "horizon": 7,
        "model": "GRU",
        "window": 90,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.20,
    },

    {
        "task": "traffic_h7",
        "target": "Total_Traffic",
        "horizon": 7,
        "model": "LSTM",
        "window": 7,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.20,
    },


    # --------------------------------------------------------
    # CASH H7
    # --------------------------------------------------------

    {
        "task": "cash_h7",
        "target": "Total_Cash",
        "horizon": 7,
        "model": "LSTM",
        "window": 7,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.20,
    },

    {
        "task": "cash_h7",
        "target": "Total_Cash",
        "horizon": 7,
        "model": "GRU",
        "window": 60,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.20,
    },
]


# ============================================================
# HELPER: MATCH ONE JSON TO ONE CANDIDATE
# ============================================================

def matches_candidate(
    config,
    candidate,
):

    return (

        config.get("input_type") == "raw"

        and

        config.get("target")
        == candidate["target"]

        and

        config.get("horizon")
        == candidate["horizon"]

        and

        config.get("model")
        == candidate["model"]

        and

        config.get("window")
        == candidate["window"]

        and

        config.get("hidden_size")
        == candidate["hidden_size"]

        and

        config.get("num_layers")
        == candidate["num_layers"]

        and

        abs(
            float(
                config.get(
                    "dropout",
                    -1
                )
            )
            -
            candidate["dropout"]
        )
        < 1e-9

        and

        config.get("seed")
        in SEEDS
    )


# ============================================================
# READ ALL METRIC JSON FILES
# ============================================================

metric_files = list(
    METRIC_DIR.glob(
        "*.json"
    )
)


if len(metric_files) == 0:

    raise RuntimeError(
        "No metric JSON files found in outputs/metrics."
    )


rows = []


# ============================================================
# FIND REQUIRED MULTI-SEED RESULTS
# ============================================================

for candidate in CANDIDATES:

    found_seeds = set()


    for json_file in metric_files:

        try:

            with open(
                json_file,
                "r"
            ) as f:

                data = json.load(
                    f
                )

        except Exception:

            continue


        config = data.get(
            "config",
            {}
        )


        if not matches_candidate(
            config,
            candidate
        ):

            continue


        validation = data.get(
            "validation"
        )


        if validation is None:

            continue


        seed = int(
            config[
                "seed"
            ]
        )


        found_seeds.add(
            seed
        )


        rows.append({

            "task":
                candidate[
                    "task"
                ],

            "target":
                candidate[
                    "target"
                ],

            "horizon":
                candidate[
                    "horizon"
                ],

            "model":
                candidate[
                    "model"
                ],

            "window":
                candidate[
                    "window"
                ],

            "hidden_size":
                candidate[
                    "hidden_size"
                ],

            "dropout":
                candidate[
                    "dropout"
                ],

            "seed":
                seed,

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


    # --------------------------------------------------------
    # Make sure seeds 1, 7, 42 all exist
    # --------------------------------------------------------

    missing_seeds = (

        set(SEEDS)
        -
        found_seeds
    )


    if missing_seeds:

        candidate_name = (

            f"{candidate['task']} | "
            f"{candidate['model']} | "
            f"W{candidate['window']}"
        )


        raise RuntimeError(

            f"\nMissing seed results for:\n"
            f"{candidate_name}\n\n"
            f"Missing seeds: "
            f"{sorted(missing_seeds)}\n"
        )


# ============================================================
# CREATE DETAILED DATAFRAME
# ============================================================

details = pd.DataFrame(
    rows
)


details = details.sort_values(

    by=[
        "task",
        "model",
        "window",
        "seed",
    ]
)


# ============================================================
# GROUP BY CANDIDATE
# ============================================================

group_columns = [

    "task",
    "target",
    "horizon",
    "model",
    "window",
    "hidden_size",
    "dropout",
]


summary = (

    details

    .groupby(
        group_columns,
        as_index=False
    )

    .agg(

        n_seeds=(
            "seed",
            "count"
        ),

        MAE_mean=(
            "MAE",
            "mean"
        ),

        MAE_std=(
            "MAE",
            "std"
        ),

        RMSE_mean=(
            "RMSE",
            "mean"
        ),

        RMSE_std=(
            "RMSE",
            "std"
        ),

        MAPE_mean=(
            "MAPE",
            "mean"
        ),

        MAPE_std=(
            "MAPE",
            "std"
        ),

        R2_mean=(
            "R2",
            "mean"
        ),

        R2_std=(
            "R2",
            "std"
        ),
    )
)


# ============================================================
# RANK WITHIN EACH TASK
#
# Primary metric = mean validation MAE
# ============================================================

summary[
    "MAE_rank"
] = (

    summary

    .groupby(
        "task"
    )[
        "MAE_mean"
    ]

    .rank(
        method="min",
        ascending=True
    )

    .astype(int)
)


summary = summary.sort_values(

    by=[
        "task",
        "MAE_rank",
    ]
)


# ============================================================
# BEST CANDIDATE PER TASK
# ============================================================

best = (

    summary[
        summary[
            "MAE_rank"
        ]
        == 1
    ]

    .copy()

    .sort_values(
        "task"
    )
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
    250
)

pd.set_option(
    "display.max_colwidth",
    None
)


# ============================================================
# PRINT INDIVIDUAL SEED RESULTS
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "RAW MULTI-SEED VALIDATION - INDIVIDUAL RUNS"
)

print(
    "=" * 120
)


for task, group in details.groupby(
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
                "seed",
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
# PRINT MEAN ± STANDARD DEVIATION
# ============================================================

print(
    "\n\n"
    + "=" * 120
)

print(
    "RAW MULTI-SEED VALIDATION SUMMARY"
)

print(
    "=" * 120
)


for task, group in summary.groupby(
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


    display_rows = []


    for _, row in group.iterrows():

        display_rows.append({

            "rank":
                row[
                    "MAE_rank"
                ],

            "model":
                row[
                    "model"
                ],

            "window":
                row[
                    "window"
                ],

            "MAE":
                (
                    f"{row['MAE_mean']:,.2f}"
                    f" ± "
                    f"{row['MAE_std']:,.2f}"
                ),

            "RMSE":
                (
                    f"{row['RMSE_mean']:,.2f}"
                    f" ± "
                    f"{row['RMSE_std']:,.2f}"
                ),

            "MAPE":
                (
                    f"{row['MAPE_mean']:.3f}%"
                    f" ± "
                    f"{row['MAPE_std']:.3f}"
                ),

            "R2":
                (
                    f"{row['R2_mean']:.4f}"
                    f" ± "
                    f"{row['R2_std']:.4f}"
                ),
        })


    display_df = pd.DataFrame(
        display_rows
    )


    print(
        display_df.to_string(
            index=False
        )
    )


# ============================================================
# PRINT BEST PER TASK
# ============================================================

print(
    "\n\n"
    + "=" * 120
)

print(
    "BEST RAW CANDIDATE PER TASK BY MEAN VALIDATION MAE"
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
            "MAE_mean",
            "MAE_std",
            "RMSE_mean",
            "MAPE_mean",
            "R2_mean",
        ]
    ]

    .to_string(
        index=False
    )
)


# ============================================================
# SAVE FILES
# ============================================================

details_path = (

    METRIC_DIR
    / "raw_multiseed_detailed.csv"
)


summary_path = (

    METRIC_DIR
    / "raw_multiseed_summary.csv"
)


best_path = (

    METRIC_DIR
    / "raw_multiseed_best.csv"
)


details.to_csv(
    details_path,
    index=False
)


summary.to_csv(
    summary_path,
    index=False
)


best.to_csv(
    best_path,
    index=False
)


print(
    "\n\nSaved:"
)

print(
    details_path
)

print(
    summary_path
)

print(
    best_path
)