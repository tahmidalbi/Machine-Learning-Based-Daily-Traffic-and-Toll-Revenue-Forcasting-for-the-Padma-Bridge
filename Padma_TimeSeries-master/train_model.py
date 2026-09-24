import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


# ============================================================
# IMPORT src/
# ============================================================

ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(ROOT / "src")
)


from src.enhanced_features import (
    prepare_timeseries_dataframe,
)

from src.sequence_data import (
    create_sequences,
)

from src.models import (
    SequenceRegressor,
)

from src.train_utils import (
    EarlyStopping,
    evaluate_loss,
    predict,
    regression_metrics,
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_indices(meta):

    target_date = pd.to_datetime(
        meta["target_date"]
    )

    # Same date ranges as the previous project.
    train_mask = (
        target_date
        <= pd.Timestamp(
            "2025-04-22"
        )
    )

    val_mask = (
        (
            target_date
            >= pd.Timestamp(
                "2025-04-23"
            )
        )
        &
        (
            target_date
            <= pd.Timestamp(
                "2025-12-22"
            )
        )
    )

    test_mask = (
        target_date
        >= pd.Timestamp(
            "2025-12-23"
        )
    )

    train_idx = np.where(
        train_mask
    )[0]

    val_idx = np.where(
        val_mask
    )[0]

    test_idx = np.where(
        test_mask
    )[0]

    return (
        train_idx,
        val_idx,
        test_idx,
    )


# ============================================================
# SCALE WITHOUT FUTURE LEAKAGE
# ============================================================

def scale_data(
    X_seq,
    X_future,
    y,
    train_idx,
    val_idx,
    test_idx,
):

    sequence_scaler = (
        StandardScaler()
    )

    target_scaler = (
        StandardScaler()
    )

    # ========================================================
    # FIT SEQUENCE SCALER ON TRAIN ONLY
    # ========================================================

    train_flat = (
        X_seq[train_idx]
        .reshape(
            -1,
            X_seq.shape[-1]
        )
    )

    sequence_scaler.fit(
        train_flat
    )

    def transform_sequences(x):

        shape = x.shape

        flat = x.reshape(
            -1,
            shape[-1]
        )

        transformed = (
            sequence_scaler
            .transform(flat)
        )

        return (
            transformed
            .reshape(shape)
            .astype(np.float32)
        )

    X_train = transform_sequences(
        X_seq[train_idx]
    )

    X_val = transform_sequences(
        X_seq[val_idx]
    )

    X_test = transform_sequences(
        X_seq[test_idx]
    )

    # ========================================================
    # FUTURE-KNOWN FEATURE SCALER
    # ========================================================

    if X_future.shape[1] > 0:

        future_scaler = (
            StandardScaler()
        )

        future_scaler.fit(
            X_future[train_idx]
        )

        F_train = (
            future_scaler
            .transform(
                X_future[train_idx]
            )
            .astype(np.float32)
        )

        F_val = (
            future_scaler
            .transform(
                X_future[val_idx]
            )
            .astype(np.float32)
        )

        F_test = (
            future_scaler
            .transform(
                X_future[test_idx]
            )
            .astype(np.float32)
        )

    else:

        future_scaler = None

        F_train = X_future[
            train_idx
        ]

        F_val = X_future[
            val_idx
        ]

        F_test = X_future[
            test_idx
        ]

    # ========================================================
    # TARGET SCALER
    # ========================================================

    target_scaler.fit(
        y[train_idx]
    )

    y_train = (
        target_scaler
        .transform(
            y[train_idx]
        )
        .astype(np.float32)
    )

    y_val = (
        target_scaler
        .transform(
            y[val_idx]
        )
        .astype(np.float32)
    )

    y_test = (
        target_scaler
        .transform(
            y[test_idx]
        )
        .astype(np.float32)
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        "F_train": F_train,
        "F_val": F_val,
        "F_test": F_test,

        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,

        "sequence_scaler":
            sequence_scaler,

        "future_scaler":
            future_scaler,

        "target_scaler":
            target_scaler,
    }


# ============================================================
# PYTORCH LOADER
# ============================================================

def make_loader(
    X,
    F,
    y,
    batch_size,
    shuffle,
):

    dataset = TensorDataset(

        torch.tensor(
            X,
            dtype=torch.float32
        ),

        torch.tensor(
            F,
            dtype=torch.float32
        ),

        torch.tensor(
            y,
            dtype=torch.float32
        ),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run(args):

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "=" * 72
    )

    print(
        "TIME-SERIES FORECASTING"
    )

    print(
        "=" * 72
    )

    print(
        f"Device       : {device}"
    )

    print(
        f"Model        : {args.model}"
    )

    print(
        f"Input type   : {args.input_type}"
    )

    print(
        f"Target       : {args.target}"
    )

    print(
        f"Window       : {args.window} days"
    )

    print(
        f"Horizon      : {args.horizon} day(s)"
    )

    print(
        f"Seed         : {args.seed}"
    )

    # ========================================================
    # DAILY MULTIVARIATE TIME SERIES
    # ========================================================

    df = (
        prepare_timeseries_dataframe()
    )

    print(
        "\nDaily dataframe shape:",
        df.shape
    )

    # ========================================================
    # CREATE ORDERED WINDOWS
    # ========================================================

    (
        X_seq,
        X_future,
        y,
        meta,
        feature_columns,

    ) = create_sequences(

        df=df,

        target=args.target,

        window=args.window,

        horizon=args.horizon,

        input_type=args.input_type,

        use_future_known=False,
    )

    print(
        "\n========================================"
    )

    print(
        "TIME-SERIES TENSOR"
    )

    print(
        "========================================"
    )

    print(
        "X sequence shape:",
        X_seq.shape
    )

    print(
        "Interpretation:"
    )

    print(
        f"{X_seq.shape[0]} samples"
    )

    print(
        f"{X_seq.shape[1]} ordered days per sample"
    )

    print(
        f"{X_seq.shape[2]} features per day"
    )

    print(
        "\nFuture-known shape:",
        X_future.shape
    )

    print(
        "Target shape:",
        y.shape
    )

    print(
        "\nSequence features:"
    )

    for i, name in enumerate(
        feature_columns,
        start=1
    ):
        print(
            f"{i:02d}. {name}"
        )

    # ========================================================
    # TIME-SERIES SANITY CHECKS
    # ========================================================

    assert len(X_seq) == len(y)
    assert len(meta) == len(y)

    assert not np.isnan(
        X_seq
    ).any()

    assert not np.isnan(
        X_future
    ).any()

    assert not np.isnan(
        y
    ).any()

    assert (
        pd.to_datetime(
            meta["target_date"]
        )
        >
        pd.to_datetime(
            meta["origin_date"]
        )
    ).all()

    gaps = (
        pd.to_datetime(
            meta["target_date"]
        )
        -
        pd.to_datetime(
            meta["origin_date"]
        )
    ).dt.days

    assert (
        gaps == args.horizon
    ).all()

    print(
        "\nTime-series sanity checks: PASSED"
    )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    (
        train_idx,
        val_idx,
        test_idx,

    ) = split_indices(
        meta
    )

    print(
        "\nChronological samples:"
    )

    print(
        "Train      :",
        len(train_idx)
    )

    print(
        "Validation :",
        len(val_idx)
    )

    print(
        "Test       :",
        len(test_idx)
    )

    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0

    print(
        "\nTrain target-date range:"
    )

    print(
        meta.iloc[
            train_idx
        ]["target_date"].min(),
        "→",
        meta.iloc[
            train_idx
        ]["target_date"].max()
    )

    print(
        "\nValidation target-date range:"
    )

    print(
        meta.iloc[
            val_idx
        ]["target_date"].min(),
        "→",
        meta.iloc[
            val_idx
        ]["target_date"].max()
    )

    print(
        "\nTest target-date range:"
    )

    print(
        meta.iloc[
            test_idx
        ]["target_date"].min(),
        "→",
        meta.iloc[
            test_idx
        ]["target_date"].max()
    )

    # ========================================================
    # SCALE — TRAIN ONLY
    # ========================================================

    scaled = scale_data(

        X_seq,
        X_future,
        y,

        train_idx,
        val_idx,
        test_idx,
    )

    # ========================================================
    # LOADERS
    # ========================================================

    train_loader = make_loader(

        scaled["X_train"],
        scaled["F_train"],
        scaled["y_train"],

        batch_size=args.batch_size,

        shuffle=True,
    )

    val_loader = make_loader(

        scaled["X_val"],
        scaled["F_val"],
        scaled["y_val"],

        batch_size=args.batch_size,

        shuffle=False,
    )

    test_loader = make_loader(

        scaled["X_test"],
        scaled["F_test"],
        scaled["y_test"],

        batch_size=args.batch_size,

        shuffle=False,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = SequenceRegressor(

        input_size=X_seq.shape[2],

        future_size=X_future.shape[1],

        model_type=args.model,

        hidden_size=args.hidden_size,

        num_layers=args.num_layers,

        dropout=args.dropout,

    ).to(device)

    print(
        "\nModel architecture:"
    )

    print(
        model
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTrainable parameters:",
        f"{total_parameters:,}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = (
        torch.nn.HuberLoss(
            delta=1.0
        )
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = (
        torch.optim.AdamW(

            model.parameters(),

            lr=args.learning_rate,

            weight_decay=
                args.weight_decay,
        )
    )

    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.5,

            patience=5,

            min_lr=1e-6,
        )
    )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stopping = (
        EarlyStopping(

            patience=
                args.patience,

            min_delta=
                1e-5,
        )
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        epoch_losses = []

        for (
            sequence,
            future,
            target

        ) in train_loader:

            sequence = (
                sequence.to(device)
            )

            future = (
                future.to(device)
            )

            target = (
                target.to(device)
            )

            optimizer.zero_grad()

            prediction = model(
                sequence,
                future
            )

            loss = criterion(
                prediction,
                target
            )

            loss.backward()

            # Recurrent gradient stabilization.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            epoch_losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(
                epoch_losses
            )
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(
            val_loss
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "val_loss"
        ].append(
            val_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(

            f"Epoch {epoch:03d} | "

            f"Train "
            f"{train_loss:.5f} | "

            f"Val "
            f"{val_loss:.5f} | "

            f"LR "
            f"{current_lr:.2e}"
        )

        stop = (
            early_stopping.step(
                val_loss,
                model
            )
        )

        if stop:

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # RESTORE BEST VALIDATION MODEL
    # ========================================================

    model.load_state_dict(
        early_stopping.best_state
    )

    print(
        "\nBest validation loss:",
        f"{early_stopping.best_loss:.6f}"
    )

    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    (
        y_val_scaled,
        y_val_pred_scaled,

    ) = predict(

        model,
        val_loader,
        device
    )

    target_scaler = scaled[
        "target_scaler"
    ]

    y_val_original = (
        target_scaler
        .inverse_transform(
            y_val_scaled
        )
    )

    y_val_pred_original = (
        target_scaler
        .inverse_transform(
            y_val_pred_scaled
        )
    )

    val_metrics = regression_metrics(

        y_val_original,
        y_val_pred_original,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "=" * 72
    )

    for key, value in val_metrics.items():

        print(
            f"{key}: "
            f"{value:.4f}"
        )

    # ========================================================
    # FINAL TEST
    # ========================================================

    test_metrics = None
    prediction_df = None

    if not args.skip_test:

        (
            y_test_scaled,
            y_pred_scaled,

        ) = predict(

            model,
            test_loader,
            device
        )

        target_scaler = scaled[
            "target_scaler"
        ]

        y_test_original = (
            target_scaler
            .inverse_transform(
                y_test_scaled
            )
        )

        y_pred_original = (
            target_scaler
            .inverse_transform(
                y_pred_scaled
            )
        )

        test_metrics = regression_metrics(

            y_test_original,
            y_pred_original,
        )

        print(
            "\n"
            + "=" * 72
        )

        print(
            "FINAL TEST RESULTS"
        )

        print(
            "=" * 72
        )

        for key, value in test_metrics.items():

            print(
                f"{key}: "
                f"{value:.4f}"
            )

    else:

        print(
            "\nTest evaluation skipped."
        )

        print(
            "Use validation metrics for model/window selection."
        )

    # ========================================================
    # EXPERIMENT NAME
    # ========================================================
    dropout_tag = str(args.dropout).replace(".", "p")
    lr_tag = f"{args.learning_rate:.0e}".replace("-", "m")
    wd_tag = f"{args.weight_decay:.0e}".replace("-", "m")

    experiment_name = (

        f"{args.model.lower()}"

        f"_{args.input_type}"

        f"_{args.target.lower()}"

        f"_w{args.window}"

        f"_h{args.horizon}"

        f"_hs{args.hidden_size}"

        f"_nl{args.num_layers}"

        f"_do{dropout_tag}"

        f"_lr{lr_tag}"

        f"_wd{wd_tag}"

        f"_seed{args.seed}"
    )
    # ========================================================
    # DIRECTORIES
    # ========================================================

    checkpoint_dir = (

        ROOT
        / "outputs"
        / "checkpoints"
        / experiment_name
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_dir = (

        ROOT
        / "outputs"
        / "predictions"
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metric_dir = (

        ROOT
        / "outputs"
        / "metrics"
    )

    metric_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    figure_dir = (

        ROOT
        / "outputs"
        / "figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    torch.save(

        {
            "model_state_dict":
                model.state_dict(),

            "input_size":
                X_seq.shape[2],

            "future_size":
                X_future.shape[1],

            "feature_columns":
                feature_columns,

            "hidden_size":
                args.hidden_size,

            "num_layers":
                args.num_layers,

            "dropout":
                args.dropout,

            "model_type":
                args.model,

            "input_type":
                args.input_type,

            "target":
                args.target,

            "window":
                args.window,

            "horizon":
                args.horizon,

            "seed":
                args.seed,
        },

        checkpoint_dir
        / "model.pt"
    )

    # ========================================================
    # SAVE SCALERS
    # ========================================================

    joblib.dump(

        scaled[
            "sequence_scaler"
        ],

        checkpoint_dir
        / "sequence_scaler.joblib"
    )

    joblib.dump(

        scaled[
            "target_scaler"
        ],

        checkpoint_dir
        / "target_scaler.joblib"
    )

    if (
        scaled[
            "future_scaler"
        ]
        is not None
    ):

        joblib.dump(

            scaled[
                "future_scaler"
            ],

            checkpoint_dir
            / "future_scaler.joblib"
        )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    all_metrics = {
        "config": {
            "model": args.model,
            "input_type": args.input_type,
            "target": args.target,

            "window": args.window,
            "horizon": args.horizon,

            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,

            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,

            "seed": args.seed,
        },

        "validation": val_metrics,

        "test": test_metrics,
    }

    with open(

        metric_dir
        / f"{experiment_name}.json",

        "w"

    ) as f:

        json.dump(
            all_metrics,
            f,
            indent=4
        )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    if not args.skip_test:

        test_meta = (

            meta
            .iloc[test_idx]
            .reset_index(
                drop=True
            )
        )

        prediction_df = (
            test_meta.copy()
        )

        prediction_df[
            "actual"
        ] = (
            y_test_original
            .reshape(-1)
        )

        prediction_df[
            "prediction"
        ] = (
            y_pred_original
            .reshape(-1)
        )

        prediction_df[
            "absolute_error"
        ] = np.abs(

            prediction_df[
                "actual"
            ]

            -

            prediction_df[
                "prediction"
            ]
        )

        prediction_df.to_csv(

            prediction_dir
            / f"{experiment_name}.csv",

            index=False
        )

    # ========================================================
    # LOSS PLOT
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        history[
            "train_loss"
        ],
        label="Training"
    )

    plt.plot(
        history[
            "val_loss"
        ],
        label="Validation"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Huber loss"
    )

    plt.title(
        f"{args.model}: "
        f"{args.input_type} "
        f"time-series training history"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        figure_dir
        / (
            f"{experiment_name}"
            "_loss.png"
        ),

        dpi=160
    )

    plt.close()
    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    if not args.skip_test:

        plt.figure(
            figsize=(14, 6)
        )

        plt.plot(

            prediction_df[
                "target_date"
            ],

            prediction_df[
                "actual"
            ],

            label="Actual"
        )

        plt.plot(

            prediction_df[
                "target_date"
            ],

            prediction_df[
                "prediction"
            ],

            label="Predicted"
        )

        plt.xlabel(
            "Date"
        )

        plt.ylabel(
            args.target
        )

        plt.title(

            f"{args.model}: "
            f"{args.input_type} time series "
            f"(window={args.window}, "
            f"horizon={args.horizon})"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(

            figure_dir
            / (
                f"{experiment_name}"
                "_predictions.png"
            ),

            dpi=160
        )

        plt.close()
# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="LSTM",
        choices=[
            "LSTM",
            "GRU",
        ],
    )

    parser.add_argument(
        "--input_type",
        type=str,
        default="enhanced",
        choices=[
            "raw",
            "enhanced",
        ],
    )

    parser.add_argument(
        "--target",
        type=str,
        default="Total_Traffic",
        choices=[
            "Total_Traffic",
            "Total_Cash",
        ],
    )

    parser.add_argument(
        "--window",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip final test evaluation during model/window tuning.",
    )
    args = parser.parse_args()

    run(args)