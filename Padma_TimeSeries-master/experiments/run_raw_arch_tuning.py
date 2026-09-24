from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


# ============================================================
# RAW ARCHITECTURE CANDIDATES
# ============================================================

CANDIDATES = [

    {
        "name": "Traffic H1 - GRU W14",
        "model": "GRU",
        "target": "Total_Traffic",
        "window": 14,
        "horizon": 1,
    },

    {
        "name": "Cash H1 - GRU W7",
        "model": "GRU",
        "target": "Total_Cash",
        "window": 7,
        "horizon": 1,
    },

    {
        "name": "Traffic H7 - LSTM W7",
        "model": "LSTM",
        "target": "Total_Traffic",
        "window": 7,
        "horizon": 7,
    },

    {
        "name": "Cash H7 - GRU W60",
        "model": "GRU",
        "target": "Total_Cash",
        "window": 60,
        "horizon": 7,
    },

    {
        "name": "Cash H7 - LSTM W7",
        "model": "LSTM",
        "target": "Total_Cash",
        "window": 7,
        "horizon": 7,
    },
]


HIDDEN_SIZES = [
    32,
    64,
    96,
]


DROPOUTS = [
    0.10,
    0.20,
    0.30,
]


# Separate tuning seed.
TUNING_SEED = 123


total_runs = (
    len(CANDIDATES)
    * len(HIDDEN_SIZES)
    * len(DROPOUTS)
)

current_run = 0


print("\n" + "=" * 80)

print(
    "STRICT RAW ARCHITECTURE TUNING"
)

print("=" * 80)

print(
    f"\nTotal runs: {total_runs}"
)

print(
    "\nTEST SET WILL NOT BE USED."
)


for candidate in CANDIDATES:

    for hidden_size in HIDDEN_SIZES:

        for dropout in DROPOUTS:

            current_run += 1

            print(
                "\n"
                + "=" * 80
            )

            print(
                f"RUN {current_run}/{total_runs}"
            )

            print(
                candidate["name"]
            )

            print(
                f"Hidden  : {hidden_size}"
            )

            print(
                f"Dropout : {dropout}"
            )

            print("=" * 80)


            command = [

                sys.executable,

                str(
                    ROOT
                    / "train_model.py"
                ),

                "--model",
                candidate["model"],

                "--input_type",
                "raw",

                "--target",
                candidate["target"],

                "--window",
                str(
                    candidate["window"]
                ),

                "--horizon",
                str(
                    candidate["horizon"]
                ),

                "--hidden_size",
                str(
                    hidden_size
                ),

                "--num_layers",
                "1",

                "--dropout",
                str(
                    dropout
                ),

                "--learning_rate",
                "0.001",

                "--weight_decay",
                "0.0001",

                "--batch_size",
                "32",

                "--seed",
                str(
                    TUNING_SEED
                ),

                "--skip_test",
            ]


            subprocess.run(
                command,
                cwd=ROOT,
                check=True,
            )


print("\n" + "=" * 80)

print(
    "RAW ARCHITECTURE TUNING COMPLETE"
)

print("=" * 80)