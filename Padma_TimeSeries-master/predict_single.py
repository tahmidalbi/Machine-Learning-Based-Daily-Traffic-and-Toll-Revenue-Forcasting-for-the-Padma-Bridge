import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


# ============================================================
# PROJECT IMPORTS
# ============================================================

ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(ROOT)
)

sys.path.insert(
    0,
    str(ROOT / "src")
)


from src.enhanced_features import (
    prepare_timeseries_dataframe,
)

from src.sequence_data import (
    get_sequence_features,
)

from src.models import (
    SequenceRegressor,
)


# ============================================================
# FINAL FROZEN RAW MODEL CONFIGURATIONS
# ============================================================

FINAL_CONFIGS = {

    # --------------------------------------------------------
    # 1-DAY TRAFFIC
    # --------------------------------------------------------
    ("Total_Traffic", 1): {
        "model": "GRU",
        "window": 14,
        "hidden_size": 96,
        "num_layers": 1,
        "dropout": 0.10,
    },

    # --------------------------------------------------------
    # 1-DAY CASH
    # --------------------------------------------------------
    ("Total_Cash", 1): {
        "model": "GRU",
        "window": 7,
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.20,
    },

    # --------------------------------------------------------
    # 7-DAY TRAFFIC
    # --------------------------------------------------------
    ("Total_Traffic", 7): {
        "model": "LSTM",
        "window": 7,
        "hidden_size": 96,
        "num_layers": 1,
        "dropout": 0.30,
    },

    # --------------------------------------------------------
    # 7-DAY CASH
    # --------------------------------------------------------
    ("Total_Cash", 7): {
        "model": "LSTM",
        "window": 7,
        "hidden_size": 96,
        "num_layers": 1,
        "dropout": 0.30,
    },
}


SEEDS = [
    1,
    7,
    42,
]


INPUT_TYPE = "raw"

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


# ============================================================
# BUILD EXPERIMENT NAME
# ============================================================

def build_experiment_name(
    model,
    target,
    window,
    horizon,
    hidden_size,
    num_layers,
    dropout,
    seed,
):

    dropout_tag = (
        str(dropout)
        .replace(".", "p")
    )

    lr_tag = (
        f"{LEARNING_RATE:.0e}"
        .replace("-", "m")
    )

    wd_tag = (
        f"{WEIGHT_DECAY:.0e}"
        .replace("-", "m")
    )

    experiment_name = (

        f"{model.lower()}"

        f"_{INPUT_TYPE}"

        f"_{target.lower()}"

        f"_w{window}"

        f"_h{horizon}"

        f"_hs{hidden_size}"

        f"_nl{num_layers}"

        f"_do{dropout_tag}"

        f"_lr{lr_tag}"

        f"_wd{wd_tag}"

        f"_seed{seed}"
    )

    return experiment_name


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    checkpoint_path,
    device,
):

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    return checkpoint


# ============================================================
# PREPARE ONE RAW HISTORICAL SEQUENCE
# ============================================================

def prepare_single_input(
    df,
    target_date,
    target,
    horizon,
    window,
):

    # --------------------------------------------------------
    # Final strict raw feature list
    # --------------------------------------------------------

    feature_columns = (
        get_sequence_features(
            "raw"
        )
    )


    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )


    target_date = pd.Timestamp(
        target_date
    )


    # --------------------------------------------------------
    # Origin date:
    #
    # H1 -> one day before target
    # H7 -> seven days before target
    # --------------------------------------------------------

    origin_date = (
        target_date
        -
        pd.Timedelta(
            days=horizon
        )
    )


    start_date = (
        origin_date
        -
        pd.Timedelta(
            days=window - 1
        )
    )


    # ========================================================
    # SELECT HISTORICAL WINDOW ONLY
    # ========================================================

    sequence_df = df[
        (
            df["Date"] >= start_date
        )
        &
        (
            df["Date"] <= origin_date
        )
    ].copy()


    sequence_df = (
        sequence_df
        .sort_values("Date")
        .reset_index(drop=True)
    )


    if len(sequence_df) != window:

        raise ValueError(

            f"\nExpected {window} historical days, "
            f"but found {len(sequence_df)}.\n\n"

            f"Required historical range:\n"
            f"{start_date.date()} -> "
            f"{origin_date.date()}\n"
        )


    # ========================================================
    # VERIFY STRICT CONSECUTIVE DAYS
    # ========================================================

    expected_dates = pd.date_range(
        start=start_date,
        end=origin_date,
        freq="D",
    )


    actual_dates = (
        sequence_df["Date"]
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )


    if not np.array_equal(
        actual_dates,
        expected_dates.to_numpy(
            dtype="datetime64[ns]"
        ),
    ):

        raise ValueError(
            "Historical sequence is not "
            "strictly consecutive daily data."
        )


    # ========================================================
    # BUILD:
    #
    # 1 x window x raw_features
    # ========================================================

    sequence = (

        sequence_df[
            feature_columns
        ]

        .to_numpy(
            dtype=np.float32
        )
    )


    if np.isnan(
        sequence
    ).any():

        raise ValueError(
            "Historical raw sequence contains NaN values."
        )


    sequence = sequence[
        np.newaxis,
        :,
        :
    ]


    # ========================================================
    # ACTUAL TARGET
    #
    # Only for comparison.
    # Never supplied to the network.
    # ========================================================

    target_rows = df[
        df["Date"] == target_date
    ]


    actual_value = None


    if len(target_rows) > 0:

        value = (
            target_rows
            .iloc[0][target]
        )

        if not pd.isna(
            value
        ):

            actual_value = float(
                value
            )


    return (
        sequence,
        actual_value,
        start_date,
        origin_date,
        target_date,
        feature_columns,
    )


# ============================================================
# PREDICT USING ONE SAVED SEED
# ============================================================

def predict_one_seed(
    sequence,
    feature_columns,
    config,
    target,
    horizon,
    seed,
    device,
):

    experiment_name = (
        build_experiment_name(

            model=
                config["model"],

            target=
                target,

            window=
                config["window"],

            horizon=
                horizon,

            hidden_size=
                config["hidden_size"],

            num_layers=
                config["num_layers"],

            dropout=
                config["dropout"],

            seed=
                seed,
        )
    )


    checkpoint_dir = (
        ROOT
        / "outputs"
        / "checkpoints"
        / experiment_name
    )


    model_path = (
        checkpoint_dir
        / "model.pt"
    )


    sequence_scaler_path = (
        checkpoint_dir
        / "sequence_scaler.joblib"
    )


    target_scaler_path = (
        checkpoint_dir
        / "target_scaler.joblib"
    )


    # --------------------------------------------------------
    # Strict raw models do NOT use future_scaler.joblib
    # --------------------------------------------------------

    required_files = [
        model_path,
        sequence_scaler_path,
        target_scaler_path,
    ]


    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(

                "\nRequired final model file "
                "was not found:\n"

                f"{file_path}\n\n"

                "Make sure the final raw training "
                "run exists for this seed."
            )


    # ========================================================
    # LOAD MODEL OBJECTS
    # ========================================================

    checkpoint = (
        load_checkpoint(
            model_path,
            device,
        )
    )


    sequence_scaler = (
        joblib.load(
            sequence_scaler_path
        )
    )


    target_scaler = (
        joblib.load(
            target_scaler_path
        )
    )


    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    if checkpoint.get(
        "input_type"
    ) != "raw":

        raise RuntimeError(
            "Checkpoint is not a raw-input model."
        )


    if checkpoint.get(
        "future_size",
        0
    ) != 0:

        raise RuntimeError(

            "This checkpoint contains future-known "
            "features, but the final project requires "
            "strict raw historical input only."
        )


    saved_features = (
        checkpoint[
            "feature_columns"
        ]
    )


    if list(
        saved_features
    ) != list(
        feature_columns
    ):

        raise RuntimeError(

            "Current raw feature order does not match "
            "the features used during training."
        )


    # ========================================================
    # SCALE SEQUENCE EXACTLY LIKE TRAINING
    # ========================================================

    original_shape = (
        sequence.shape
    )


    sequence_flat = (
        sequence.reshape(
            -1,
            sequence.shape[-1]
        )
    )


    sequence_scaled = (

        sequence_scaler

        .transform(
            sequence_flat
        )

        .reshape(
            original_shape
        )

        .astype(
            np.float32
        )
    )


    # ========================================================
    # REBUILD SAVED MODEL
    # ========================================================

    model = SequenceRegressor(

        input_size=
            checkpoint[
                "input_size"
            ],

        future_size=
            checkpoint[
                "future_size"
            ],

        model_type=
            checkpoint[
                "model_type"
            ],

        hidden_size=
            checkpoint[
                "hidden_size"
            ],

        num_layers=
            checkpoint[
                "num_layers"
            ],

        dropout=
            checkpoint[
                "dropout"
            ],

    ).to(
        device
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    sequence_tensor = (

        torch.tensor(
            sequence_scaled,
            dtype=torch.float32,
        )

        .to(
            device
        )
    )


    # ========================================================
    # INFERENCE
    #
    # No future tensor.
    # No future-known features.
    # No optimizer.
    # No backward.
    # ========================================================

    with torch.no_grad():

        prediction_scaled = model(
            sequence_tensor
        )


    prediction_scaled = (
        prediction_scaled
        .cpu()
        .numpy()
    )


    prediction = (

        target_scaler

        .inverse_transform(
            prediction_scaled
        )

        .reshape(-1)[0]
    )


    return float(
        prediction
    )


# ============================================================
# MAIN
# ============================================================

def main(args):

    # --------------------------------------------------------
    # Accept both short and full target names
    # --------------------------------------------------------

    if args.target in [
        "traffic",
        "Total_Traffic",
    ]:

        target = (
            "Total_Traffic"
        )

        target_short = (
            "traffic"
        )


    elif args.target in [
        "cash",
        "Total_Cash",
    ]:

        target = (
            "Total_Cash"
        )

        target_short = (
            "cash"
        )


    else:

        raise ValueError(
            "Target must be traffic or cash."
        )


    key = (
        target,
        args.horizon,
    )


    if key not in FINAL_CONFIGS:

        raise ValueError(
            "Only horizons 1 and 7 are supported."
        )


    config = (
        FINAL_CONFIGS[key]
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "PADMA BRIDGE FINAL RAW FORECAST"
    )

    print(
        "=" * 72
    )


    print(
        f"\nDevice       : {device}"
    )

    print(
        f"Input type   : RAW"
    )

    print(
        f"Target       : {target}"
    )

    print(
        f"Model        : {config['model']}"
    )

    print(
        f"Window       : {config['window']} days"
    )

    print(
        f"Horizon      : {args.horizon} day(s)"
    )

    print(
        f"Hidden size  : {config['hidden_size']}"
    )

    print(
        f"Dropout      : {config['dropout']}"
    )

    print(
        f"Forecast date: {args.date}"
    )


    # ========================================================
    # PREPARE DATAFRAME
    # ========================================================

    df = (
        prepare_timeseries_dataframe()
    )


    (
        sequence,
        actual_value,
        start_date,
        origin_date,
        target_date,
        feature_columns,

    ) = prepare_single_input(

        df=df,

        target_date=
            args.date,

        target=
            target,

        horizon=
            args.horizon,

        window=
            config[
                "window"
            ],
    )


    # ========================================================
    # DISPLAY INPUT INFORMATION
    # ========================================================

    print(
        "\nINPUT SEQUENCE"
    )

    print(
        "-" * 72
    )


    print(
        f"Starts       : {start_date.date()}"
    )

    print(
        f"Ends         : {origin_date.date()}"
    )

    print(
        f"Ordered days : {sequence.shape[1]}"
    )

    print(
        f"Features/day : {sequence.shape[2]}"
    )

    print(
        f"Tensor shape : {sequence.shape}"
    )


    print(
        "\nRaw features:"
    )

    for feature in feature_columns:

        print(
            f"  - {feature}"
        )


    print(
        "\nImportant:"
    )

    print(
        "Only historical raw observations are "
        "supplied to the model."
    )

    print(
        "No lag features, rolling averages, "
        "or target-date future-known features are used."
    )

    print(
        f"No observation after "
        f"{origin_date.date()} "
        f"is used as model input."
    )


    # ========================================================
    # THREE-SEED PREDICTIONS
    # ========================================================

    predictions = []


    print(
        "\nSEED PREDICTIONS"
    )

    print(
        "-" * 72
    )


    for seed in SEEDS:

        prediction = (
            predict_one_seed(

                sequence=
                    sequence,

                feature_columns=
                    feature_columns,

                config=
                    config,

                target=
                    target,

                horizon=
                    args.horizon,

                seed=
                    seed,

                device=
                    device,
            )
        )


        predictions.append(
            prediction
        )


        print(
            f"Seed {seed:>2}: "
            f"{prediction:,.2f}"
        )


    # ========================================================
    # ENSEMBLE
    # ========================================================

    ensemble_prediction = float(
        np.mean(
            predictions
        )
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL 3-SEED ENSEMBLE FORECAST"
    )

    print(
        "=" * 72
    )


    if target == "Total_Traffic":

        print(
            f"\nPredicted traffic : "
            f"{ensemble_prediction:,.0f} vehicles"
        )

    else:

        print(
            f"\nPredicted toll    : "
            f"{ensemble_prediction:,.2f} BDT"
        )


    # ========================================================
    # ACTUAL VALUE — DISPLAY ONLY
    # ========================================================

    if actual_value is not None:

        absolute_error = abs(
            actual_value
            -
            ensemble_prediction
        )


        percentage_error = (

            absolute_error
            /
            abs(actual_value)
            *
            100
        )


        print(
            "\nACTUAL VALUE "
            "(evaluation only)"
        )

        print(
            "-" * 72
        )


        if target == "Total_Traffic":

            print(
                f"Actual traffic    : "
                f"{actual_value:,.0f} vehicles"
            )

            print(
                f"Absolute error    : "
                f"{absolute_error:,.0f} vehicles"
            )

        else:

            print(
                f"Actual toll       : "
                f"{actual_value:,.2f} BDT"
            )

            print(
                f"Absolute error    : "
                f"{absolute_error:,.2f} BDT"
            )


        print(
            f"Percentage error  : "
            f"{percentage_error:.2f}%"
        )


    else:

        print(
            "\nActual target value is not available."
        )

        print(
            "Prediction was still produced using "
            "the available historical sequence."
        )


    # ========================================================
    # DATA SPLIT
    # ========================================================

    if (
        target_date
        <= pd.Timestamp(
            "2025-04-22"
        )
    ):

        split_name = "TRAIN"


    elif (
        target_date
        <= pd.Timestamp(
            "2025-12-22"
        )
    ):

        split_name = "VALIDATION"


    elif (
        target_date
        <= df["Date"].max()
    ):

        split_name = "TEST"


    else:

        split_name = "FUTURE"


    print(
        "\nDATA SPLIT"
    )

    print(
        "-" * 72
    )

    print(
        f"Target date category: "
        f"{split_name}"
    )


    # ========================================================
    # SAVE PREDICTION
    # ========================================================

    output_dir = (
        ROOT
        / "outputs"
        / "live_predictions"
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    output_file = (

        output_dir

        /

        (
            f"{target_short}"
            f"_h{args.horizon}"
            f"_{target_date.date()}"
            f".csv"
        )
    )


    row = {

        "target":
            target,

        "input_type":
            "raw",

        "model":
            config["model"],

        "window":
            config["window"],

        "horizon":
            args.horizon,

        "hidden_size":
            config["hidden_size"],

        "dropout":
            config["dropout"],

        "input_start_date":
            start_date.date(),

        "input_end_date":
            origin_date.date(),

        "forecast_date":
            target_date.date(),

        "seed_1_prediction":
            predictions[0],

        "seed_7_prediction":
            predictions[1],

        "seed_42_prediction":
            predictions[2],

        "ensemble_prediction":
            ensemble_prediction,

        "actual":
            actual_value,

        "split":
            split_name,
    }


    pd.DataFrame(
        [row]
    ).to_csv(
        output_file,
        index=False
    )


    print(
        "\nSaved prediction:"
    )

    print(
        output_file
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--target",

        required=True,

        choices=[
            "traffic",
            "cash",
            "Total_Traffic",
            "Total_Cash",
        ],
    )


    parser.add_argument(

        "--horizon",

        required=True,

        type=int,

        choices=[
            1,
            7,
        ],
    )


    parser.add_argument(

        "--date",

        required=True,

        type=str,

        help=(
            "Forecast target date "
            "in YYYY-MM-DD format."
        ),
    )


    args = parser.parse_args()

    main(
        args
    )