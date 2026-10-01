# %%
"""
post_train_analysis.py

Runs expensive post-training analyses on one saved checkpoint:

    1. Analytic baseline
    2. 1024-shot evaluation
    3. 4096-shot evaluation
    4. 8192-shot evaluation
    5. Resource/specification analysis

Usage:

    python post_train_analysis.py \
        ../results/phase2a/Phase 2A - Random Trainable/seed_42/checkpoint.pt
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import pennylane as qml


PROJECT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_DIR not in sys.path:
    sys.path.append(
        PROJECT_DIR
    )

from models.hybrid_classifier import (
    HybridQConvClassifier
)

from data.mnist_preprocessing import (
    get_mnist_datasets
)


def _make_test_subset(
    test_set,
    n_samples
):

    n_samples = min(
        int(n_samples),
        len(test_set)
    )

    return torch.utils.data.Subset(
        test_set,
        range(n_samples)
    )


def _evaluate_model(
    model,
    dataset
):

    model.eval()

    predictions = []
    targets = []
    feature_maps = []

    with torch.no_grad():

        for index in range(
            len(dataset)
        ):

            image, label = (
                dataset[index]
            )

            logit, feature_map = (
                model(
                    image.unsqueeze(0),
                    return_feature_map=True
                )
            )

            probability = (
                torch.sigmoid(
                    logit
                ).item()
            )

            prediction = (
                1
                if probability >= 0.5
                else 0
            )

            predictions.append(
                prediction
            )

            targets.append(
                int(label)
            )

            feature_maps.append(
                feature_map.detach().cpu()
            )

    return (
        np.asarray(predictions),
        np.asarray(targets),
        torch.stack(feature_maps)
    )


def run_shot_robustness(
    model,
    test_set,
    n_samples=100
):

    # ---------------------------------------------------------------
    # Analytic reference
    # ---------------------------------------------------------------

    model.set_shots(
        None
    )

    (
        analytic_predictions,
        analytic_targets,
        analytic_features
    ) = _evaluate_model(
        model,
        test_set
    )

    results = {}

    # ---------------------------------------------------------------
    # Finite-shot settings
    # ---------------------------------------------------------------

    for shots in [
        1024,
        4096,
        8192
    ]:

        model.set_shots(
            shots
        )

        (
            shot_predictions,
            shot_targets,
            shot_features
        ) = _evaluate_model(
            model,
            test_set
        )

        difference = (
            shot_features
            - analytic_features
        )

        results[
            f"shots_{shots}"
        ] = {

            "accuracy": float(
                accuracy_score(
                    shot_targets,
                    shot_predictions
                )
            ),

            "f1": float(
                f1_score(
                    shot_targets,
                    shot_predictions,
                    zero_division=0
                )
            ),

            "mae_vs_analytic": float(
                difference
                .abs()
                .mean()
                .item()
            ),

            "rmse_vs_analytic": float(
                torch.sqrt(
                    (
                        difference
                        ** 2
                    ).mean()
                ).item()
            ),

            "max_error_vs_analytic":
                float(
                    difference
                    .abs()
                    .max()
                    .item()
                ),
        }

    # Restore analytic execution.
    model.set_shots(
        None
    )

    return results


def run_resource_analysis(
    model
):

    model.set_shots(
        None
    )

    final_kernels = (
        model.kernels
        .detach()
    )

    norms = torch.linalg.norm(
        final_kernels,
        dim=1,
        keepdim=True
    )

    safe_norms = torch.where(
        norms > 1e-9,
        norms,
        torch.ones_like(norms)
    )

    normalized_kernels = (
        final_kernels
        / safe_norms
    )

    # Every patch must be a valid StatePrep state.
    dummy_patches = torch.zeros(
        model.E * model.F,
        model.R * model.S,
        dtype=normalized_kernels.dtype
    )

    dummy_patches[:, 0] = 1.0

    specs = qml.specs(
        model.qnode
    )(
        normalized_kernels,
        dummy_patches
    )

    safe_specs = {}

    for key, value in (
        specs.items()
    ):

        try:
            json.dumps(
                value
            )

            safe_specs[
                key
            ] = value

        except TypeError:

            safe_specs[
                key
            ] = str(value)

    return safe_specs


def run_from_checkpoint(
    checkpoint_path,
    test_samples=500,
    shot_samples=100
):

    checkpoint_path = os.path.abspath(
        checkpoint_path
    )

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    config = checkpoint[
        "config"
    ]

    # ---------------------------------------------------------------
    # Rebuild model
    # ---------------------------------------------------------------

    model = HybridQConvClassifier(
        M=config["M"],
        R=config["R"],
        S=config["S"],
        E=config["E"],
        F=config["F"],
        init_mode=config[
            "init_mode"
        ]
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # ---------------------------------------------------------------
    # Load test set
    # ---------------------------------------------------------------

    _, _, test_set = (
        get_mnist_datasets(
            seed=config["seed"]
        )
    )

    test_set = _make_test_subset(
        test_set,
        test_samples
    )

    # ---------------------------------------------------------------
    # Shot robustness
    # ---------------------------------------------------------------

    shot_subset = _make_test_subset(
        test_set,
        shot_samples
    )

    shot_results = (
        run_shot_robustness(
            model,
            shot_subset,
            n_samples=shot_samples
        )
    )

    # ---------------------------------------------------------------
    # Resource analysis
    # ---------------------------------------------------------------

    resource_results = (
        run_resource_analysis(
            model
        )
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    out_dir = os.path.dirname(
        checkpoint_path
    )

    with open(
        os.path.join(
            out_dir,
            "shot_robustness.json"
        ),
        "w"
    ) as f:

        json.dump(
            shot_results,
            f,
            indent=4
        )

    with open(
        os.path.join(
            out_dir,
            "resource_summary.json"
        ),
        "w"
    ) as f:

        json.dump(
            resource_results,
            f,
            indent=4
        )

    print(
        "\n--- Shot Robustness ---"
    )

    print(
        json.dumps(
            shot_results,
            indent=4
        )
    )

    print(
        "\n--- Resource Summary ---"
    )

    print(
        json.dumps(
            resource_results,
            indent=4
        )
    )

    return (
        shot_results,
        resource_results
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:

        raise SystemExit(
            "Usage:\n"
            "python post_train_analysis.py "
            "<checkpoint.pt>"
        )

    run_from_checkpoint(
        sys.argv[1]
    )