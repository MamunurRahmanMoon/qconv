# %%
"""
trainer.py

Phase-2A training and validation engine.

Responsibilities:
    1. Train the existing Seed-QConv model.
    2. Check gradient numerical integrity BEFORE clipping.
    3. Never replace NaN/Inf gradients with artificial values.
    4. Measure raw and normalized kernel movement.
    5. Measure functional feature-map movement.
    6. Verify classical-vs-quantum QConv correctness.
    7. Save checkpoints, metrics, logs, and figures.
    8. Keep expensive finite-shot/resource analysis optional.

Important:
    Phase-2A currently uses batch_size=1 because the existing
    QConv QNode/training loop is single-sample.
"""

import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import pennylane as qml


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from models.hybrid_classifier import HybridQConvClassifier
from data.mnist_preprocessing import get_mnist_datasets


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _make_subset(dataset, n_samples):
    """
    Return the first n_samples from a dataset.

    If n_samples is None, return the full dataset.
    """
    if n_samples is None:
        return dataset

    n_samples = int(n_samples)

    if n_samples >= len(dataset):
        return dataset

    return torch.utils.data.Subset(
        dataset,
        range(n_samples)
    )


def _safe_normalize_rows(x, eps=1e-9):
    """
    Row-wise L2 normalization.

    Used only for detached analysis, not for the differentiable
    model forward pass.
    """
    norms = torch.linalg.norm(
        x,
        dim=1,
        keepdim=True
    )

    safe_norms = torch.where(
        norms > eps,
        norms,
        torch.ones_like(norms)
    )

    return x / safe_norms, norms


def get_gradient_stats(model):
    """
    Inspect gradients of all trainable parameters.

    Returns
    -------
    all_finite : bool
        True only if every existing trainable gradient is finite.

    stats : dict
        Per-parameter gradient statistics.
    """

    all_finite = True
    stats = {}

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if parameter.grad is None:

            stats[name] = {
                "finite": False,
                "missing": True,
                "norm": None,
                "max_abs": None,
            }

            all_finite = False
            continue

        grad = parameter.grad.detach()

        finite = bool(
            torch.isfinite(grad).all().item()
        )

        stats[name] = {
            "finite": finite,
            "missing": False,
            "norm": (
                float(
                    torch.linalg.norm(grad).item()
                )
                if finite else None
            ),
            "max_abs": (
                float(
                    grad.abs().max().item()
                )
                if finite else None
            ),
        }

        if not finite:
            all_finite = False

    return all_finite, stats


def _save_gradient_failure(
    out_dir,
    epoch,
    step,
    grad_stats,
    model,
):
    """
    Save complete diagnostic information before stopping.
    """

    failure = {
        "epoch": int(epoch + 1),
        "step": int(step + 1),
        "gradient_stats": grad_stats,
        "kernel_values": model.kernels.detach().cpu().tolist(),
    }

    path = os.path.join(
        out_dir,
        f"nonfinite_gradient_epoch_{epoch+1}_step_{step+1}.json"
    )

    with open(path, "w") as f:
        json.dump(
            failure,
            f,
            indent=4
        )

    return path


def _evaluate(
    model,
    loader,
    criterion,
):
    """
    Evaluate one dataset split.
    """

    model.eval()

    losses = []
    predictions = []
    targets = []

    with torch.no_grad():

        for images, labels in loader:

            if images.shape[0] != 1:
                raise ValueError(
                    "Phase-2A requires batch_size=1."
                )

            image = images[0]
            label = labels[0].float()

            logit = model(
                image.unsqueeze(0)
            )

            loss = criterion(
                logit,
                label
            )

            probability = torch.sigmoid(
                logit
            ).item()

            prediction = (
                1
                if probability >= 0.5
                else 0
            )

            losses.append(
                float(loss.item())
            )

            predictions.append(
                prediction
            )

            targets.append(
                int(label.item())
            )

    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float(
            accuracy_score(
                targets,
                predictions
            )
        ),
        "precision": float(
            precision_score(
                targets,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                targets,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                targets,
                predictions,
                zero_division=0
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                targets,
                predictions
            ).tolist()
        ),
    }

    return metrics


def _collect_feature_maps(
    model,
    dataset,
    n_samples,
):
    """
    Calculate QConv feature maps for a fixed subset.
    """

    model.eval()

    n = min(
        int(n_samples),
        len(dataset)
    )

    feature_maps = []
    labels = []

    with torch.no_grad():

        for index in range(n):

            image, label = dataset[index]

            _, feature_map = model(
                image.unsqueeze(0),
                return_feature_map=True
            )

            feature_maps.append(
                feature_map.detach().cpu()
            )

            labels.append(
                int(label)
            )

    return (
        torch.stack(feature_maps),
        labels,
    )


def _extract_raw_patches(
    image,
    R,
    S,
):
    """
    Extract raw, unnormalized image patches.

    Returns
    -------
    patches : [E*F, R*S]
    E
    F
    """

    image_2d = image.squeeze()

    height, width = image_2d.shape

    E = height - R + 1
    F = width - S + 1

    patches = []

    for i in range(E):

        for j in range(F):

            patch = image_2d[
                i:i + R,
                j:j + S
            ].flatten()

            patches.append(
                patch
            )

    return (
        torch.stack(patches),
        E,
        F,
    )


def _classical_vs_quantum_check(
    model,
    dataset,
    n_samples,
):
    """
    Compare the trained quantum reconstruction with:

        Y_classical = K @ X^T

    using the final raw kernels.
    """

    model.eval()

    n = min(
        int(n_samples),
        len(dataset)
    )

    final_kernels = (
        model.kernels
        .detach()
        .cpu()
    )

    mae_values = []
    rmse_values = []
    max_values = []

    with torch.no_grad():

        for index in range(n):

            image, _ = dataset[index]

            raw_patches, E, F = (
                _extract_raw_patches(
                    image,
                    model.R,
                    model.S,
                )
            )

            classical = torch.matmul(
                final_kernels,
                raw_patches.T
            )

            classical = classical.view(
                model.M,
                E,
                F
            )

            _, quantum = model(
                image.unsqueeze(0),
                return_feature_map=True
            )

            quantum = quantum.detach().cpu()

            difference = (
                quantum
                - classical
            )

            mae_values.append(
                float(
                    difference.abs()
                    .mean()
                    .item()
                )
            )

            rmse_values.append(
                float(
                    torch.sqrt(
                        (difference ** 2)
                        .mean()
                    ).item()
                )
            )

            max_values.append(
                float(
                    difference.abs()
                    .max()
                    .item()
                )
            )

    return {
        "c_vs_q_mae": float(
            np.mean(mae_values)
        ),
        "c_vs_q_rmse": float(
            np.mean(rmse_values)
        ),
        "c_vs_q_max": float(
            np.max(max_values)
        ),
    }


def _save_training_figures(
    out_dir,
    epoch_history,
):
    """
    Save training/validation loss and accuracy figures.
    """

    if len(epoch_history) == 0:
        return

    epochs = [
        item["epoch"]
        for item in epoch_history
    ]

    train_loss = [
        item["train_loss"]
        for item in epoch_history
    ]

    val_loss = [
        item["val_loss"]
        for item in epoch_history
    ]

    train_accuracy = [
        item["train_accuracy"]
        for item in epoch_history
    ]

    val_accuracy = [
        item["val_accuracy"]
        for item in epoch_history
    ]

    # Loss
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_loss,
        marker="o",
        label="Train Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(
        "Phase 2A Training / Validation Loss"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "training_loss.png"
        ),
        dpi=180
    )

    plt.close()

    # Accuracy
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_accuracy,
        marker="o",
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Phase 2A Training / Validation Accuracy"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "training_accuracy.png"
        ),
        dpi=180
    )

    plt.close()


def _save_kernel_figure(
    out_dir,
    initial_kernels,
    final_kernels,
):
    """
    Save initial/final normalized kernel visualizations.
    """

    initial_norm, _ = _safe_normalize_rows(
        initial_kernels
    )

    final_norm, _ = _safe_normalize_rows(
        final_kernels
    )

    M = initial_norm.shape[0]

    fig, axes = plt.subplots(
        M,
        2,
        figsize=(7, 3.5 * M)
    )

    if M == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )

    for m in range(M):

        axes[m, 0].imshow(
            initial_norm[m]
            .reshape(4, 4),
            aspect="auto"
        )

        axes[m, 0].set_title(
            f"Kernel {m} - Initial"
        )

        axes[m, 0].set_xlabel("Column")
        axes[m, 0].set_ylabel("Row")

        axes[m, 1].imshow(
            final_norm[m]
            .reshape(4, 4),
            aspect="auto"
        )

        axes[m, 1].set_title(
            f"Kernel {m} - Final"
        )

        axes[m, 1].set_xlabel("Column")
        axes[m, 1].set_ylabel("Row")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "kernel_initial_vs_final.png"
        ),
        dpi=180
    )

    plt.close()


def _save_feature_figure(
    out_dir,
    y_initial,
    y_final,
):
    """
    Save the first sample's initial/final feature maps
    and absolute difference.

    Expected shape:
        [N, M, E, F]
    """

    if y_initial.shape[0] == 0:
        return

    initial = y_initial[0]
    final = y_final[0]
    difference = (
        final - initial
    ).abs()

    M = initial.shape[0]

    fig, axes = plt.subplots(
        M,
        3,
        figsize=(9, 3 * M)
    )

    if M == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )

    for m in range(M):

        axes[m, 0].imshow(
            initial[m],
            aspect="auto"
        )

        axes[m, 0].set_title(
            f"Kernel {m} - Initial Feature"
        )

        axes[m, 1].imshow(
            final[m],
            aspect="auto"
        )

        axes[m, 1].set_title(
            f"Kernel {m} - Final Feature"
        )

        axes[m, 2].imshow(
            difference[m],
            aspect="auto"
        )

        axes[m, 2].set_title(
            f"Kernel {m} - Absolute Difference"
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "feature_map_initial_final_difference.png"
        ),
        dpi=180
    )

    plt.close()


# ---------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------

def run_experiment(config):

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    exp_name = config.get(
        "exp_name",
        "experiment"
    )

    init_mode = config.get(
        "init_mode",
        "random"
    )

    freeze_qconv = bool(
        config.get(
            "freeze_qconv",
            False
        )
    )

    M = int(
        config.get(
            "M",
            2
        )
    )

    R = int(
        config.get(
            "R",
            4
        )
    )

    S = int(
        config.get(
            "S",
            R
        )
    )

    E = int(
        config.get(
            "E",
            4
        )
    )

    F = int(
        config.get(
            "F",
            E
        )
    )

    epochs = int(
        config.get(
            "epochs",
            2
        )
    )

    lr = float(
        config.get(
            "lr",
            0.01
        )
    )

    seed = int(
        config.get(
            "seed",
            42
        )
    )

    batch_size = int(
        config.get(
            "batch_size",
            1
        )
    )

    gradient_clipping = bool(
        config.get(
            "gradient_clipping",
            True
        )
    )

    max_grad_norm = float(
        config.get(
            "max_grad_norm",
            1.0
        )
    )

    debug_numerics = bool(
        config.get(
            "debug_numerics",
            True
        )
    )

    fail_on_nonfinite_grad = bool(
        config.get(
            "fail_on_nonfinite_grad",
            True
        )
    )

    train_samples = config.get(
        "train_samples",
        None
    )

    val_samples = config.get(
        "val_samples",
        None
    )

    test_samples = config.get(
        "test_samples",
        None
    )

    analyze_features = bool(
        config.get(
            "analyze_features",
            True
        )
    )

    feature_eval_samples = int(
        config.get(
            "feature_eval_samples",
            20
        )
    )

    run_shot_eval = bool(
        config.get(
            "run_shot_eval",
            False
        )
    )

    run_resource_analysis = bool(
        config.get(
            "run_resource_analysis",
            False
        )
    )

    # ---------------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------------

    if batch_size != 1:
        raise ValueError(
            "Phase-2A currently requires batch_size=1."
        )

    if M < 1:
        raise ValueError(
            "M must be >= 1."
        )

    # ---------------------------------------------------------------
    # Deterministic seeds
    # ---------------------------------------------------------------

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(
        "\n"
        + "=" * 70
        + f"\nStarting: {exp_name}"
        + f"\nSeed: {seed}"
        + f"\nTrainable QConv: {not freeze_qconv}"
        + f"\nInitialization: {init_mode}"
        + "\n"
        + "=" * 70
    )

    # ---------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------

    out_dir = os.path.abspath(
        os.path.join(
            PROJECT_DIR,
            "results",
            "phase2a",
            exp_name,
            f"seed_{seed}"
        )
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    train_set, val_set, test_set = (
        get_mnist_datasets(
            seed=seed
        )
    )

    train_set = _make_subset(
        train_set,
        train_samples
    )

    val_set = _make_subset(
        val_set,
        val_samples
    )

    test_set = _make_subset(
        test_set,
        test_samples
    )

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Train samples: {len(train_set)}"
    )

    print(
        f"Validation samples: {len(val_set)}"
    )

    print(
        f"Test samples: {len(test_set)}"
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model = HybridQConvClassifier(
        M=M,
        R=R,
        S=S,
        E=E,
        F=F,
        init_mode=init_mode
    )

    model.kernels.requires_grad = (
        not freeze_qconv
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=lr
    )

    # ---------------------------------------------------------------
    # Initial kernels
    # ---------------------------------------------------------------

    initial_kernels = (
        model.kernels
        .detach()
        .clone()
    )

    initial_normalized, initial_norms = (
        _safe_normalize_rows(
            initial_kernels
        )
    )

    # ---------------------------------------------------------------
    # Initial feature maps
    # ---------------------------------------------------------------

    feature_eval_count = min(
        feature_eval_samples,
        len(test_set)
    )

    if analyze_features and feature_eval_count > 0:

        Y_initial, feature_labels = (
            _collect_feature_maps(
                model,
                test_set,
                feature_eval_count
            )
        )

    else:

        Y_initial = None
        feature_labels = []

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    history = []
    epoch_history = []

    non_finite_events = 0

    start_time = time.time()

    for epoch in range(epochs):

        model.train()

        train_losses = []
        train_predictions = []
        train_targets = []

        for step, (
            images,
            labels
        ) in enumerate(train_loader):

            if images.shape[0] != 1:
                raise RuntimeError(
                    "Phase-2A requires batch_size=1."
                )

            image = images[0]

            label = labels[0].float()

            optimizer.zero_grad(
                set_to_none=True
            )

            # ---------------------------------------------------
            # Forward
            # ---------------------------------------------------

            logit = model(
                image.unsqueeze(0)
            )

            if (
                debug_numerics
                and not torch.isfinite(
                    logit
                ).all()
            ):
                raise RuntimeError(
                    "Non-finite logit detected "
                    f"at epoch={epoch + 1}, "
                    f"step={step + 1}."
                )

            loss = criterion(
                logit,
                label
            )

            if (
                debug_numerics
                and not torch.isfinite(
                    loss
                ).all()
            ):
                raise RuntimeError(
                    "Non-finite loss detected "
                    f"at epoch={epoch + 1}, "
                    f"step={step + 1}."
                )

            # ---------------------------------------------------
            # Backward
            # ---------------------------------------------------

            loss.backward()

            # ---------------------------------------------------
            # GRADIENT DIAGNOSTICS
            #
            # IMPORTANT:
            # We NEVER call torch.nan_to_num().
            # ---------------------------------------------------

            all_grads_finite, grad_stats = (
                get_gradient_stats(
                    model
                )
            )

            step_log = {
                "epoch": epoch + 1,
                "step": step + 1,
                "loss": float(
                    loss.item()
                ),
                "all_grads_finite": (
                    all_grads_finite
                ),
            }

            for parameter_name, stats in (
                grad_stats.items()
            ):

                step_log[
                    f"{parameter_name}_grad_finite"
                ] = stats["finite"]

                step_log[
                    f"{parameter_name}_grad_norm"
                ] = stats["norm"]

                step_log[
                    f"{parameter_name}_grad_max_abs"
                ] = stats["max_abs"]

            # ---------------------------------------------------
            # Stop on non-finite gradients
            # ---------------------------------------------------

            if not all_grads_finite:

                non_finite_events += 1

                failure_path = (
                    _save_gradient_failure(
                        out_dir,
                        epoch,
                        step,
                        grad_stats,
                        model
                    )
                )

                message = (
                    "Non-finite gradient detected.\n"
                    f"Epoch: {epoch + 1}\n"
                    f"Step: {step + 1}\n"
                    f"Diagnostic saved to: "
                    f"{failure_path}"
                )

                if fail_on_nonfinite_grad:
                    raise RuntimeError(
                        message
                    )

            # ---------------------------------------------------
            # Gradient clipping
            #
            # ONLY applied if gradients are finite.
            # ---------------------------------------------------

            if gradient_clipping:

                pre_clip_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        model.parameters(),
                        max_norm=max_grad_norm
                    )
                )

                pre_clip_norm = float(
                    pre_clip_norm.item()
                )

                if not math.isfinite(
                    pre_clip_norm
                ):
                    raise RuntimeError(
                        "Total gradient norm "
                        "was non-finite before "
                        "clipping."
                    )

                step_log[
                    "pre_clip_total_norm"
                ] = pre_clip_norm

                if (
                    model.kernels.requires_grad
                    and model.kernels.grad
                    is not None
                ):

                    step_log[
                        "post_clip_kernel_grad_norm"
                    ] = float(
                        torch.linalg.norm(
                            model.kernels.grad
                        ).item()
                    )

            # ---------------------------------------------------
            # Optimizer
            # ---------------------------------------------------

            optimizer.step()

            # ---------------------------------------------------
            # Parameter finite check
            # ---------------------------------------------------

            if debug_numerics:

                for name, parameter in (
                    model.named_parameters()
                ):

                    if (
                        parameter.requires_grad
                        and not torch.isfinite(
                            parameter
                        ).all()
                    ):

                        raise RuntimeError(
                            f"Non-finite parameter "
                            f"'{name}' detected "
                            "after optimizer step."
                        )

            # ---------------------------------------------------
            # Training metrics
            # ---------------------------------------------------

            probability = torch.sigmoid(
                logit
            ).item()

            prediction = (
                1
                if probability >= 0.5
                else 0
            )

            train_losses.append(
                float(loss.item())
            )

            train_predictions.append(
                prediction
            )

            train_targets.append(
                int(label.item())
            )

            history.append(
                step_log
            )

            if (
                step + 1
            ) % 100 == 0:

                print(
                    f"Epoch {epoch + 1} | "
                    f"Step {step + 1} | "
                    f"Loss {loss.item():.4f}"
                )

        # -------------------------------------------------------
        # Epoch training metrics
        # -------------------------------------------------------

        train_loss = float(
            np.mean(
                train_losses
            )
        )

        train_accuracy = float(
            accuracy_score(
                train_targets,
                train_predictions
            )
        )

        # -------------------------------------------------------
        # Validation
        # -------------------------------------------------------

        val_metrics = _evaluate(
            model,
            val_loader,
            criterion
        )

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics[
                "accuracy"
            ],
            "val_precision": val_metrics[
                "precision"
            ],
            "val_recall": val_metrics[
                "recall"
            ],
            "val_f1": val_metrics[
                "f1"
            ],
        }

        epoch_history.append(
            epoch_row
        )

        print(
            "\n"
            f"Epoch {epoch + 1}/{epochs}\n"
            f"Train Loss: {train_loss:.6f}\n"
            f"Train Acc : {train_accuracy:.6f}\n"
            f"Val Loss  : {val_metrics['loss']:.6f}\n"
            f"Val Acc   : {val_metrics['accuracy']:.6f}\n"
        )

    # ---------------------------------------------------------------
    # Final test evaluation
    # ---------------------------------------------------------------

    print(
        "\n--- Final Test Evaluation ---"
    )

    test_metrics = _evaluate(
        model,
        test_loader,
        criterion
    )

    print(
        f"Test Loss: "
        f"{test_metrics['loss']:.6f}"
    )

    print(
        f"Test Acc: "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"Test F1: "
        f"{test_metrics['f1']:.6f}"
    )

    # ---------------------------------------------------------------
    # Final kernels
    # ---------------------------------------------------------------

    final_kernels = (
        model.kernels
        .detach()
        .clone()
    )

    final_normalized, final_norms = (
        _safe_normalize_rows(
            final_kernels
        )
    )

    raw_kernel_delta = float(
        torch.linalg.norm(
            final_kernels
            - initial_kernels
        ).item()
    )

    normalized_kernel_delta = {}

    kernel_cosine = {}

    for m in range(M):

        normalized_kernel_delta[
            f"m{m}"
        ] = float(
            torch.linalg.norm(
                final_normalized[m]
                - initial_normalized[m]
            ).item()
        )

        kernel_cosine[
            f"m{m}"
        ] = float(
            torch.dot(
                initial_normalized[m],
                final_normalized[m]
            ).item()
        )

    # ---------------------------------------------------------------
    # Feature-map movement
    # ---------------------------------------------------------------

    if (
        analyze_features
        and Y_initial is not None
    ):

        Y_final, _ = (
            _collect_feature_maps(
                model,
                test_set,
                feature_eval_count
            )
        )

        feature_difference = (
            Y_final
            - Y_initial
        )

        feature_mae = float(
            feature_difference
            .abs()
            .mean()
            .item()
        )

        feature_rmse = float(
            torch.sqrt(
                (
                    feature_difference
                    ** 2
                ).mean()
            ).item()
        )

        feature_max_abs_difference = float(
            feature_difference
            .abs()
            .max()
            .item()
        )

        relative_feature_change = float(
            (
                torch.linalg.norm(
                    feature_difference
                )
                /
                (
                    torch.linalg.norm(
                        Y_initial
                    )
                    + 1e-12
                )
            ).item()
        )

        np.savez(
            os.path.join(
                out_dir,
                "feature_maps.npz"
            ),
            y_initial=Y_initial.numpy(),
            y_final=Y_final.numpy(),
            labels=np.asarray(
                feature_labels
            )
        )

        _save_feature_figure(
            out_dir,
            Y_initial,
            Y_final
        )

    else:

        feature_mae = None
        feature_rmse = None
        feature_max_abs_difference = None
        relative_feature_change = None

    # ---------------------------------------------------------------
    # Classical-vs-quantum correctness
    # ---------------------------------------------------------------

    correctness = (
        _classical_vs_quantum_check(
            model,
            test_set,
            feature_eval_count
        )
    )

    print(
        "\n--- Classical vs Quantum Correctness ---"
    )

    print(
        f"MAE  : "
        f"{correctness['c_vs_q_mae']:.12e}"
    )

    print(
        f"RMSE : "
        f"{correctness['c_vs_q_rmse']:.12e}"
    )

    print(
        f"MAX  : "
        f"{correctness['c_vs_q_max']:.12e}"
    )

    # ---------------------------------------------------------------
    # Runtime
    # ---------------------------------------------------------------

    runtime_seconds = (
        time.time()
        - start_time
    )

    # ---------------------------------------------------------------
    # Save model checkpoint
    # ---------------------------------------------------------------

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "initial_kernels":
            initial_kernels.cpu(),

        "final_kernels":
            final_kernels.cpu(),

        "config": {
            "exp_name": exp_name,
            "init_mode": init_mode,
            "freeze_qconv": freeze_qconv,
            "M": M,
            "R": R,
            "S": S,
            "E": E,
            "F": F,
            "epochs": epochs,
            "lr": lr,
            "seed": seed,
        },
    }

    torch.save(
        checkpoint,
        os.path.join(
            out_dir,
            "checkpoint.pt"
        )
    )

    # ---------------------------------------------------------------
    # Save metrics JSON
    # ---------------------------------------------------------------

    metrics = {
        "exp_name": exp_name,
        "seed": seed,

        "init_mode": init_mode,
        "freeze_qconv": freeze_qconv,

        "M": M,
        "R": R,
        "S": S,
        "E": E,
        "F": F,

        "epochs": epochs,
        "lr": lr,

        "train_samples": len(
            train_set
        ),
        "val_samples": len(
            val_set
        ),
        "test_samples": len(
            test_set
        ),

        "test": test_metrics,

        "raw_kernel_delta":
            raw_kernel_delta,

        "normalized_kernel_delta":
            normalized_kernel_delta,

        "kernel_cosine":
            kernel_cosine,

        "initial_kernel_norms":
            initial_norms.flatten()
            .tolist(),

        "final_kernel_norms":
            final_norms.flatten()
            .tolist(),

        "feature_mae":
            feature_mae,

        "feature_rmse":
            feature_rmse,

        "feature_max_abs_difference":
            feature_max_abs_difference,

        "relative_feature_change":
            relative_feature_change,

        "c_vs_q_mae":
            correctness[
                "c_vs_q_mae"
            ],

        "c_vs_q_rmse":
            correctness[
                "c_vs_q_rmse"
            ],

        "c_vs_q_max":
            correctness[
                "c_vs_q_max"
            ],

        "non_finite_gradient_events":
            non_finite_events,

        "runtime_seconds":
            runtime_seconds,
    }

    with open(
        os.path.join(
            out_dir,
            "metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # ---------------------------------------------------------------
    # Save step logs
    # ---------------------------------------------------------------

    if history:

        with open(
            os.path.join(
                out_dir,
                "step_logs.csv"
            ),
            "w",
            newline=""
        ) as f:

            fieldnames = sorted({
                key
                for row in history
                for key in row.keys()
            })

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore"
            )

            writer.writeheader()

            writer.writerows(
                history
            )

    # ---------------------------------------------------------------
    # Save epoch history
    # ---------------------------------------------------------------

    if epoch_history:

        with open(
            os.path.join(
                out_dir,
                "epoch_history.csv"
            ),
            "w",
            newline=""
        ) as f:

            fieldnames = list(
                epoch_history[0].keys()
            )

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                epoch_history
            )

    # ---------------------------------------------------------------
    # Save kernel history
    # ---------------------------------------------------------------

    torch.save(
        {
            "initial_kernels":
                initial_kernels.cpu(),

            "final_kernels":
                final_kernels.cpu(),

            "initial_normalized":
                initial_normalized.cpu(),

            "final_normalized":
                final_normalized.cpu(),
        },
        os.path.join(
            out_dir,
            "kernel_history.pt"
        )
    )

    # ---------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------

    _save_training_figures(
        out_dir,
        epoch_history
    )

    _save_kernel_figure(
        out_dir,
        initial_kernels,
        final_kernels
    )

    # ---------------------------------------------------------------
    # Optional expensive analysis
    # ---------------------------------------------------------------

    if run_shot_eval:

        print(
            "\nRunning shot robustness analysis..."
        )

        from post_train_analysis import (
            run_shot_robustness
        )

        shot_results = (
            run_shot_robustness(
                model,
                test_set,
                n_samples=min(
                    100,
                    len(test_set)
                )
            )
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

    if run_resource_analysis:

        print(
            "\nRunning resource analysis..."
        )

        model.set_shots(
            None
        )

        try:

            dummy_patches = torch.zeros(
                E * F,
                R * S,
                dtype=final_normalized.dtype
            )

            # Every state-preparation input must be valid.
            dummy_patches[:, 0] = 1.0

            specs = qml.specs(
                model.qnode
            )(
                final_normalized,
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

            with open(
                os.path.join(
                    out_dir,
                    "resource_summary.json"
                ),
                "w"
            ) as f:

                json.dump(
                    safe_specs,
                    f,
                    indent=4
                )

        except Exception as exc:

            with open(
                os.path.join(
                    out_dir,
                    "resource_analysis_error.txt"
                ),
                "w"
            ) as f:

                f.write(
                    repr(exc)
                )

    # ---------------------------------------------------------------
    # Final console summary
    # ---------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
        + "\nExperiment Complete"
        + f"\nOutput: {out_dir}"
        + f"\nTest Accuracy: {test_metrics['accuracy']:.6f}"
        + f"\nTest F1: {test_metrics['f1']:.6f}"
        + f"\nTest Loss: {test_metrics['loss']:.6f}"
        + f"\nRaw Kernel Delta: {raw_kernel_delta:.6f}"
        + f"\nNormalized Kernel Delta: {normalized_kernel_delta}"
        + f"\nRelative Feature Change: {relative_feature_change}"
        + f"\nC-vs-Q MAE: {correctness['c_vs_q_mae']:.12e}"
        + f"\nRuntime: {runtime_seconds:.2f} seconds"
        + "\n"
        + "=" * 70
    )

    return metrics