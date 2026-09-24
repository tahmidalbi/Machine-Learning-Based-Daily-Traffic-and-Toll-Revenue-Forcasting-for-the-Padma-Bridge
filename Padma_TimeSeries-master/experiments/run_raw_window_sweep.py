from pathlib import Path
import subprocess
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent


# ============================================================
# WINDOW CANDIDATES
# ============================================================

WINDOWS = [
    7,
    14,
    30,
    60,
    90,
]


# ============================================================
# MODEL TYPES
# ============================================================

MODELS = [
    "LSTM",
    "GRU",
]


# ============================================================
# FOUR FORECASTING TASKS
# ============================================================

TASKS = [

    {
        "name": "Traffic H1",
        "target": "Total_Traffic",
        "horizon": 1,
    },

    {
        "name": "Cash H1",
        "target": "Total_Cash",
        "horizon": 1,
    },

    {
        "name": "Traffic H7",
        "target": "Total_Traffic",
        "horizon": 7,
    },

    {
        "name": "Cash H7",
        "target": "Total_Cash",
        "horizon": 7,
    },
]


# ============================================================
# NEUTRAL ARCHITECTURE FOR WINDOW SELECTION
# ============================================================

HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.20

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

BATCH_SIZE = 32


# ------------------------------------------------------------
# Use a separate seed so old seed 1/7/42 files are untouched.
# ------------------------------------------------------------

SWEEP_SEED = 123


# ============================================================
# TOTAL RUN COUNT
# ============================================================

total_runs = (
    len(TASKS)
    * len(MODELS)
    * len(WINDOWS)
)

current_run = 0


print(
    "\n"
    + "=" * 80
)

print(
    "STRICT RAW TIME-SERIES WINDOW SWEEP"
)

print(
    "=" * 80
)

print(
    f"\nTotal runs: {total_runs}"
)

print(
    f"Seed: {SWEEP_SEED}"
)

print(
    "\nTEST SET WILL NOT BE USED."
)

print(
    "Window selection uses validation metrics only."
)


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

for task in TASKS:

    print(
        "\n\n"
        + "#" * 80
    )

    print(
        task["name"]
    )

    print(
        "#" * 80
    )


    for model in MODELS:

        for window in WINDOWS:

            current_run += 1


            print(
                "\n"
                + "=" * 80
            )

            print(
                f"RUN {current_run}/{total_runs}"
            )

            print(
                f"Task    : {task['name']}"
            )

            print(
                f"Model   : {model}"
            )

            print(
                f"Window  : {window}"
            )

            print(
                f"Horizon : {task['horizon']}"
            )

            print(
                "=" * 80
            )


            command = [

                sys.executable,

                str(
                    ROOT
                    / "train_model.py"
                ),

                "--model",
                model,

                "--input_type",
                "raw",

                "--target",
                task["target"],

                "--window",
                str(window),

                "--horizon",
                str(
                    task["horizon"]
                ),

                "--hidden_size",
                str(
                    HIDDEN_SIZE
                ),

                "--num_layers",
                str(
                    NUM_LAYERS
                ),

                "--dropout",
                str(
                    DROPOUT
                ),

                "--learning_rate",
                str(
                    LEARNING_RATE
                ),

                "--weight_decay",
                str(
                    WEIGHT_DECAY
                ),

                "--batch_size",
                str(
                    BATCH_SIZE
                ),

                "--seed",
                str(
                    SWEEP_SEED
                ),

                # IMPORTANT:
                # validation only
                "--skip_test",
            ]


            subprocess.run(

                command,

                cwd=ROOT,

                check=True,
            )


print(
    "\n\n"
    + "=" * 80
)

print(
    "RAW WINDOW SWEEP COMPLETE"
)

print(
    "=" * 80
)

print(
    f"\nCompleted {total_runs} runs."
)

print(
    "\nNow run:"
)

print(
    "python summarize_raw_window_sweep.py"
)