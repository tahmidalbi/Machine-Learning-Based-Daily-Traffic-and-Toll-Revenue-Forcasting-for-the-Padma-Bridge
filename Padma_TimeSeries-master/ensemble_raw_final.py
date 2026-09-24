from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "outputs"

PREDICTION_DIR = (
    OUTPUT_DIR
    / "predictions"
)

METRIC_DIR = (
    OUTPUT_DIR
    / "metrics"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)


PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FINAL RAW MODELS
# ============================================================

FINAL_MODELS = {

    # ========================================================
    # TRAFFIC - 1 DAY
    # ========================================================

    "traffic_h1": {

        "experiment_base":
            "gru_raw_total_traffic_"
            "w14_h1_hs96_nl1_do0p1_"
            "lr1em03_wd1em04",

        "ensemble_filename":
            "gru_raw_total_traffic_"
            "w14_h1_seed_ensemble.csv",

        "metric_filename":
            "traffic_h1_raw_final.json",

        "figure_filename":
            "traffic_h1_raw_final.png",

        "title":
            "Raw GRU - Traffic 1 Day Ahead",

        "ylabel":
            "Total Traffic",
    },


    # ========================================================
    # CASH - 1 DAY
    # ========================================================

    "cash_h1": {

        "experiment_base":
            "gru_raw_total_cash_"
            "w7_h1_hs64_nl1_do0p2_"
            "lr1em03_wd1em04",

        "ensemble_filename":
            "gru_raw_total_cash_"
            "w7_h1_seed_ensemble.csv",

        "metric_filename":
            "cash_h1_raw_final.json",

        "figure_filename":
            "cash_h1_raw_final.png",

        "title":
            "Raw GRU - Toll 1 Day Ahead",

        "ylabel":
            "Total Cash (BDT)",
    },


    # ========================================================
    # TRAFFIC - 7 DAYS
    # ========================================================

    "traffic_h7": {

        "experiment_base":
            "lstm_raw_total_traffic_"
            "w7_h7_hs96_nl1_do0p3_"
            "lr1em03_wd1em04",

        "ensemble_filename":
            "lstm_raw_total_traffic_"
            "w7_h7_seed_ensemble.csv",

        "metric_filename":
            "traffic_h7_raw_final.json",

        "figure_filename":
            "traffic_h7_raw_final.png",

        "title":
            "Raw LSTM - Traffic 7 Days Ahead",

        "ylabel":
            "Total Traffic",
    },


    # ========================================================
    # CASH - 7 DAYS
    # ========================================================

    "cash_h7": {

        "experiment_base":
            "lstm_raw_total_cash_"
            "w7_h7_hs96_nl1_do0p3_"
            "lr1em03_wd1em04",

        "ensemble_filename":
            "lstm_raw_total_cash_"
            "w7_h7_seed_ensemble.csv",

        "metric_filename":
            "cash_h7_raw_final.json",

        "figure_filename":
            "cash_h7_raw_final.png",

        "title":
            "Raw LSTM - Toll 7 Days Ahead",

        "ylabel":
            "Total Cash (BDT)",
    },
}


SEEDS = [
    1,
    7,
    42,
]


# ============================================================
# PROCESS ONE MODEL
# ============================================================

def create_ensemble(
    short_name,
    config,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        short_name.upper()
    )

    print(
        "=" * 80
    )


    experiment_base = (
        config[
            "experiment_base"
        ]
    )


    # ========================================================
    # FIND THE THREE SEED FILES
    # ========================================================

    seed_files = []


    for seed in SEEDS:

        filename = (
            f"{experiment_base}"
            f"_seed{seed}.csv"
        )

        file_path = (
            PREDICTION_DIR
            / filename
        )


        if not file_path.exists():

            raise FileNotFoundError(

                "\nMissing prediction file:\n"
                f"{file_path}\n\n"
                "Make sure this raw model was trained "
                "WITHOUT --skip_test."
            )


        seed_files.append(
            file_path
        )


    print(
        "\nPrediction files:"
    )


    for path in seed_files:

        print(
            path.name
        )


    # ========================================================
    # LOAD BASE DATA
    # ========================================================

    first = pd.read_csv(
        seed_files[0]
    )


    first[
        "target_date"
    ] = pd.to_datetime(
        first[
            "target_date"
        ]
    )


    ensemble = first[
        [
            "origin_date",
            "target_date",
            "actual",
        ]
    ].copy()


    # ========================================================
    # ADD EACH SEED PREDICTION
    # ========================================================

    for seed, file_path in zip(
        SEEDS,
        seed_files,
    ):

        current = pd.read_csv(
            file_path
        )


        current[
            "target_date"
        ] = pd.to_datetime(
            current[
                "target_date"
            ]
        )


        # ----------------------------------------------------
        # SAME DATES?
        # ----------------------------------------------------

        if not np.array_equal(

            ensemble[
                "target_date"
            ].values,

            current[
                "target_date"
            ].values,

        ):

            raise RuntimeError(

                f"Target dates do not match "
                f"in {file_path.name}"
            )


        # ----------------------------------------------------
        # SAME ACTUAL VALUES?
        # ----------------------------------------------------

        if not np.allclose(

            ensemble[
                "actual"
            ].values,

            current[
                "actual"
            ].values,

        ):

            raise RuntimeError(

                f"Actual values do not match "
                f"in {file_path.name}"
            )


        ensemble[
            f"prediction_seed_{seed}"
        ] = (

            current[
                "prediction"
            ].values
        )


    # ========================================================
    # THREE-SEED ENSEMBLE
    # ========================================================

    prediction_columns = [

        "prediction_seed_1",
        "prediction_seed_7",
        "prediction_seed_42",
    ]


    ensemble[
        "ensemble_prediction"
    ] = (

        ensemble[
            prediction_columns
        ]

        .mean(
            axis=1
        )
    )


    # ========================================================
    # ERROR PER TEST SAMPLE
    # ========================================================

    ensemble[
        "absolute_error"
    ] = np.abs(

        ensemble[
            "actual"
        ]

        -

        ensemble[
            "ensemble_prediction"
        ]
    )


    ensemble[
        "percentage_error"
    ] = (

        ensemble[
            "absolute_error"
        ]

        /

        np.abs(
            ensemble[
                "actual"
            ]
        )

        *

        100
    )


    # ========================================================
    # FINAL TEST METRICS
    # ========================================================

    y_true = (

        ensemble[
            "actual"
        ]
        .to_numpy()
    )


    y_pred = (

        ensemble[
            "ensemble_prediction"
        ]
        .to_numpy()
    )


    mae = mean_absolute_error(
        y_true,
        y_pred,
    )


    rmse = np.sqrt(

        mean_squared_error(
            y_true,
            y_pred,
        )
    )


    mape = (

        mean_absolute_percentage_error(
            y_true,
            y_pred,
        )

        *

        100
    )


    r2 = r2_score(
        y_true,
        y_pred,
    )


    metrics = {

        "model":
            config[
                "title"
            ],

        "input_type":
            "raw",

        "number_of_test_samples":
            int(
                len(
                    ensemble
                )
            ),

        "MAE":
            float(
                mae
            ),

        "RMSE":
            float(
                rmse
            ),

        "MAPE":
            float(
                mape
            ),

        "R2":
            float(
                r2
            ),
    }


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\nFINAL RAW ENSEMBLE TEST RESULTS"
    )

    print(
        "-" * 60
    )


    print(
        f"Test samples : "
        f"{len(ensemble)}"
    )


    print(
        f"MAE          : "
        f"{mae:,.4f}"
    )


    print(
        f"RMSE         : "
        f"{rmse:,.4f}"
    )


    print(
        f"MAPE         : "
        f"{mape:.4f}%"
    )


    print(
        f"R2           : "
        f"{r2:.4f}"
    )


    # ========================================================
    # SAVE FINAL ENSEMBLE CSV
    # ========================================================

    ensemble_path = (

        PREDICTION_DIR

        /

        config[
            "ensemble_filename"
        ]
    )


    ensemble.to_csv(
        ensemble_path,
        index=False,
    )


    # ========================================================
    # SAVE FINAL METRICS JSON
    # ========================================================

    metric_path = (

        METRIC_DIR

        /

        config[
            "metric_filename"
        ]
    )


    with open(
        metric_path,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )


    # ========================================================
    # FINAL ACTUAL VS PREDICTED GRAPH
    # ========================================================

    plt.figure(
        figsize=(14, 6)
    )


    plt.plot(

        ensemble[
            "target_date"
        ],

        ensemble[
            "actual"
        ],

        label="Actual",
    )


    plt.plot(

        ensemble[
            "target_date"
        ],

        ensemble[
            "ensemble_prediction"
        ],

        label="Predicted",
    )


    plt.xlabel(
        "Date"
    )


    plt.ylabel(
        config[
            "ylabel"
        ]
    )


    plt.title(
        config[
            "title"
        ]
        +
        " - 3-Seed Ensemble"
    )


    plt.legend()

    plt.tight_layout()


    figure_path = (

        FIGURE_DIR

        /

        config[
            "figure_filename"
        ]
    )


    plt.savefig(
        figure_path,
        dpi=180,
    )


    plt.close()


    # ========================================================
    # PRINT FILES
    # ========================================================

    print(
        "\nSaved ensemble:"
    )

    print(
        ensemble_path
    )


    print(
        "\nSaved metrics:"
    )

    print(
        metric_path
    )


    print(
        "\nSaved figure:"
    )

    print(
        figure_path
    )


    return metrics


# ============================================================
# RUN ALL FOUR FINAL MODELS
# ============================================================

all_results = {}


for short_name, config in FINAL_MODELS.items():

    all_results[
        short_name
    ] = create_ensemble(

        short_name,
        config,
    )


# ============================================================
# CREATE ONE SUMMARY CSV
# ============================================================

summary_rows = []


for name, metrics in all_results.items():

    summary_rows.append({

        "task":
            name,

        "input_type":
            "raw",

        "MAE":
            metrics[
                "MAE"
            ],

        "RMSE":
            metrics[
                "RMSE"
            ],

        "MAPE":
            metrics[
                "MAPE"
            ],

        "R2":
            metrics[
                "R2"
            ],

        "test_samples":
            metrics[
                "number_of_test_samples"
            ],
    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_path = (

    METRIC_DIR

    /

    "raw_final_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False,
)


print(
    "\n"
    + "=" * 80
)

print(
    "ALL RAW ENSEMBLES COMPLETE"
)

print(
    "=" * 80
)


print(
    "\nFINAL SUMMARY"
)


print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\nSummary saved to:"
)

print(
    summary_path
)