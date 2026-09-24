from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


# ============================================================
# FINAL FROZEN RAW TIME-SERIES CONFIGURATIONS
#
# These four configurations were selected through:
#   1. Window sweep  (run_raw_window_sweep.py)
#   2. Architecture tuning  (run_raw_arch_tuning.py)
#   3. Multi-seed stability check  (summarize_raw_multiseed.py)
#
# Do NOT change these values — they define the final project.
# ============================================================

CONFIGS = [

    # --------------------------------------------------------
    # TRAFFIC H1
    # GRU  |  W14  |  HS 96  |  DO 0.10
    # --------------------------------------------------------
    {
        "name": "Traffic H1",
        "model": "GRU",
        "target": "Total_Traffic",
        "window": 14,
        "horizon": 1,
        "hidden_size": 96,
        "dropout": 0.10,
    },

    # --------------------------------------------------------
    # CASH H1
    # GRU  |  W7   |  HS 64  |  DO 0.20
    # --------------------------------------------------------
    {
        "name": "Cash H1",
        "model": "GRU",
        "target": "Total_Cash",
        "window": 7,
        "horizon": 1,
        "hidden_size": 64,
        "dropout": 0.20,
    },

    # --------------------------------------------------------
    # TRAFFIC H7
    # LSTM |  W7   |  HS 96  |  DO 0.30
    # --------------------------------------------------------
    {
        "name": "Traffic H7",
        "model": "LSTM",
        "target": "Total_Traffic",
        "window": 7,
        "horizon": 7,
        "hidden_size": 96,
        "dropout": 0.30,
    },

    # --------------------------------------------------------
    # CASH H7
    # LSTM |  W7   |  HS 96  |  DO 0.30
    # --------------------------------------------------------
    {
        "name": "Cash H7",
        "model": "LSTM",
        "target": "Total_Cash",
        "window": 7,
        "horizon": 7,
        "hidden_size": 96,
        "dropout": 0.30,
    },
]


SEEDS = [
    1,
    7,
    42,
]


# ============================================================
# TRAIN EVERYTHING
# 4 configurations × 3 seeds = 12 training runs total
# ============================================================

total_runs = (
    len(CONFIGS)
    * len(SEEDS)
)

run_number = 0


for config in CONFIGS:

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"STARTING: {config['name']}"
    )

    print(
        "=" * 80
    )

    for seed in SEEDS:

        run_number += 1

        print(
            f"\nRUN {run_number}/{total_runs}"
        )

        print(
            f"{config['name']} | Seed {seed}"
        )

        command = [

            sys.executable,

            str(
                ROOT
                / "train_model.py"
            ),

            "--model",
            config["model"],

            "--input_type",
            "raw",

            "--target",
            config["target"],

            "--window",
            str(
                config["window"]
            ),

            "--horizon",
            str(
                config["horizon"]
            ),

            "--hidden_size",
            str(
                config["hidden_size"]
            ),

            "--num_layers",
            "1",

            "--dropout",
            str(
                config["dropout"]
            ),

            "--learning_rate",
            "0.001",

            "--weight_decay",
            "0.0001",

            "--seed",
            str(seed),
        ]

        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
        )


print(
    "\n"
    + "=" * 80
)

print(
    "ALL FINAL RAW MODELS FINISHED"
)

print(
    "=" * 80
)

print(
    f"\nCompleted {total_runs} training runs."
)

print(
    "\nOutputs are in:"
)

print(
    ROOT / "outputs"
)