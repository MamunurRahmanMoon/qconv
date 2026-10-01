# %%
"""
aggregate_results.py

Collect all completed Phase-2A runs and generate:

    phase2a_per_seed.csv
    phase2a_summary.csv

Figures:
    figure_accuracy.png
    figure_test_loss.png
    figure_feature_change.png
    figure_classical_quantum_mae.png
    figure_validation_loss.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = (
    Path(__file__).resolve()
    .parents[1]
    / "results"
    / "phase2a"
)

OUT = (
    ROOT
    / "aggregate"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------------------
# Collect metrics
# ---------------------------------------------------------------------

rows = []


for metrics_path in ROOT.glob(
    "*/*/metrics.json"
):

    with open(
        metrics_path
    ) as f:

        metrics = json.load(
            f
        )

    row = {

        "experiment":
            metrics["exp_name"],

        "seed":
            metrics["seed"],

        "test_loss":
            metrics["test"]["loss"],

        "test_accuracy":
            metrics["test"]["accuracy"],

        "test_precision":
            metrics["test"]["precision"],

        "test_recall":
            metrics["test"]["recall"],

        "test_f1":
            metrics["test"]["f1"],

        "raw_kernel_delta":
            metrics[
                "raw_kernel_delta"
            ],

        "feature_mae":
            metrics.get(
                "feature_mae",
                np.nan
            ),

        "feature_rmse":
            metrics.get(
                "feature_rmse",
                np.nan
            ),

        "feature_max_abs_difference":
            metrics.get(
                "feature_max_abs_difference",
                np.nan
            ),

        "relative_feature_change":
            metrics.get(
                "relative_feature_change",
                np.nan
            ),

        "c_vs_q_mae":
            metrics.get(
                "c_vs_q_mae",
                np.nan
            ),

        "c_vs_q_rmse":
            metrics.get(
                "c_vs_q_rmse",
                np.nan
            ),

        "c_vs_q_max":
            metrics.get(
                "c_vs_q_max",
                np.nan
            ),

        "non_finite_gradient_events":
            metrics.get(
                "non_finite_gradient_events",
                np.nan
            ),

        "runtime_seconds":
            metrics.get(
                "runtime_seconds",
                np.nan
            ),
    }

    # Add normalized kernel metrics.
    for kernel_name, value in (
        metrics[
            "normalized_kernel_delta"
        ].items()
    ):

        row[
            f"normalized_kernel_delta_{kernel_name}"
        ] = value

    # Add cosine similarities.
    for kernel_name, value in (
        metrics[
            "kernel_cosine"
        ].items()
    ):

        row[
            f"kernel_cosine_{kernel_name}"
        ] = value

    rows.append(
        row
    )


if not rows:

    raise RuntimeError(
        "No metrics.json files were found."
    )


df = pd.DataFrame(
    rows
)


# ---------------------------------------------------------------------
# Save per-seed table
# ---------------------------------------------------------------------

df.to_csv(
    OUT
    / "phase2a_per_seed.csv",
    index=False
)


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

summary = (
    df.groupby(
        "experiment"
    )
    .agg(

        n_runs=(
            "seed",
            "count"
        ),

        test_accuracy_mean=(
            "test_accuracy",
            "mean"
        ),

        test_accuracy_std=(
            "test_accuracy",
            "std"
        ),

        test_f1_mean=(
            "test_f1",
            "mean"
        ),

        test_f1_std=(
            "test_f1",
            "std"
        ),

        test_loss_mean=(
            "test_loss",
            "mean"
        ),

        test_loss_std=(
            "test_loss",
            "std"
        ),

        raw_kernel_delta_mean=(
            "raw_kernel_delta",
            "mean"
        ),

        feature_rmse_mean=(
            "feature_rmse",
            "mean"
        ),

        relative_feature_change_mean=(
            "relative_feature_change",
            "mean"
        ),

        c_vs_q_mae_mean=(
            "c_vs_q_mae",
            "mean"
        ),

        c_vs_q_rmse_mean=(
            "c_vs_q_rmse",
            "mean"
        ),

        c_vs_q_max_mean=(
            "c_vs_q_max",
            "mean"
        ),

        non_finite_gradient_events=(
            "non_finite_gradient_events",
            "sum"
        ),

        runtime_mean_seconds=(
            "runtime_seconds",
            "mean"
        ),
    )
    .reset_index()
)


summary.to_csv(
    OUT
    / "phase2a_summary.csv",
    index=False
)


# ---------------------------------------------------------------------
# Figure 1: Test accuracy
# ---------------------------------------------------------------------

x = np.arange(
    len(summary)
)

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    x,
    summary[
        "test_accuracy_mean"
    ],
    yerr=summary[
        "test_accuracy_std"
    ].fillna(0),
    capsize=4
)

plt.xticks(
    x,
    summary[
        "experiment"
    ],
    rotation=25,
    ha="right"
)

plt.ylabel(
    "Test Accuracy"
)

plt.title(
    "Phase 2A Test Accuracy"
)

plt.tight_layout()

plt.savefig(
    OUT
    / "figure_accuracy.png",
    dpi=180
)

plt.close()


# ---------------------------------------------------------------------
# Figure 2: Test loss
# ---------------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    x,
    summary[
        "test_loss_mean"
    ],
    yerr=summary[
        "test_loss_std"
    ].fillna(0),
    capsize=4
)

plt.xticks(
    x,
    summary[
        "experiment"
    ],
    rotation=25,
    ha="right"
)

plt.ylabel(
    "Test BCE Loss"
)

plt.title(
    "Phase 2A Test Loss"
)

plt.tight_layout()

plt.savefig(
    OUT
    / "figure_test_loss.png",
    dpi=180
)

plt.close()


# ---------------------------------------------------------------------
# Figure 3: Functional feature-map change
# ---------------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    x,
    summary[
        "relative_feature_change_mean"
    ]
)

plt.xticks(
    x,
    summary[
        "experiment"
    ],
    rotation=25,
    ha="right"
)

plt.ylabel(
    "Relative Feature-Map Change"
)

plt.title(
    "Functional QConv Feature-Map Change"
)

plt.tight_layout()

plt.savefig(
    OUT
    / "figure_feature_change.png",
    dpi=180
)

plt.close()


# ---------------------------------------------------------------------
# Figure 4: Classical vs quantum correctness
# ---------------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    x,
    summary[
        "c_vs_q_mae_mean"
    ]
)

plt.xticks(
    x,
    summary[
        "experiment"
    ],
    rotation=25,
    ha="right"
)

plt.ylabel(
    "Classical vs Quantum MAE"
)

plt.title(
    "Seed-QConv Mathematical Correctness"
)

plt.tight_layout()

plt.savefig(
    OUT
    / "figure_classical_quantum_mae.png",
    dpi=180
)

plt.close()


# ---------------------------------------------------------------------
# Figure 5: Validation-loss curves
# ---------------------------------------------------------------------

plt.figure(
    figsize=(11, 7)
)

for history_path in ROOT.glob(
    "*/*/epoch_history.csv"
):

    history = pd.read_csv(
        history_path
    )

    experiment_name = (
        history_path
        .parent
        .parent
        .name
    )

    seed_name = (
        history_path
        .parent
        .name
    )

    label = (
        f"{experiment_name} | "
        f"{seed_name}"
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        marker="o",
        label=label
    )


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Validation BCE Loss"
)

plt.title(
    "Phase 2A Validation-Loss Curves"
)

plt.legend(
    fontsize=7
)

plt.tight_layout()

plt.savefig(
    OUT
    / "figure_validation_loss.png",
    dpi=180
)

plt.close()


# ---------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------

print(
    "\n"
    + "=" * 80
    + "\nPHASE 2A SUMMARY"
    + "\n"
    + "=" * 80
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nSaved:"
)

print(
    OUT
    / "phase2a_per_seed.csv"
)

print(
    OUT
    / "phase2a_summary.csv"
)

print(
    OUT
    / "figure_accuracy.png"
)

print(
    OUT
    / "figure_test_loss.png"
)

print(
    OUT
    / "figure_feature_change.png"
)

print(
    OUT
    / "figure_classical_quantum_mae.png"
)

print(
    OUT
    / "figure_validation_loss.png"
)