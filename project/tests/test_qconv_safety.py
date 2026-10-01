# %%
"""
test_qconv_safety.py

Run from project/:

    python -m pytest tests/test_qconv_safety.py -q
"""

import torch

from data.mnist_preprocessing import (
    extract_and_normalize_patches,
)

from models.hybrid_classifier import (
    HybridQConvClassifier,
)


def test_zero_patch_is_safe():

    image = torch.zeros(
        7,
        7,
        dtype=torch.float32
    )

    patches, norms, E, F = (
        extract_and_normalize_patches(
            image,
            R=4,
            S=4
        )
    )

    assert E == 4
    assert F == 4

    # Every patch is zero.
    assert torch.allclose(
        norms,
        torch.zeros_like(norms)
    )

    # The state sent toward StatePrep must be finite.
    assert torch.isfinite(
        patches
    ).all()

    # Each zero patch is represented by |0...0>.
    assert torch.allclose(
        patches[:, 0],
        torch.ones(16)
    )

    assert torch.all(
        patches[:, 1:] == 0
    )


def test_zero_kernel_is_safe():

    model = HybridQConvClassifier(
        M=2,
        R=4,
        S=4,
        E=4,
        F=4,
        init_mode="random"
    )

    with torch.no_grad():

        model.kernels[0].zero_()

    image = torch.zeros(
        1,
        7,
        7,
        dtype=torch.float32
    )

    logit, feature_map = (
        model(
            image,
            return_feature_map=True
        )
    )

    assert torch.isfinite(
        logit
    ).all()

    assert torch.isfinite(
        feature_map
    ).all()

    # Zero kernel AND zero image must give zero contribution.
    assert torch.allclose(
        feature_map[0],
        torch.zeros_like(
            feature_map[0]
        ),
        atol=1e-7
    )


def test_forward_backward_has_finite_gradient():

    model = HybridQConvClassifier(
        M=2,
        R=4,
        S=4,
        E=4,
        F=4,
        init_mode="random"
    )

    image = torch.rand(
        1,
        7,
        7
    )

    target = torch.tensor(
        1.0
    )

    logit = model(
        image
    )

    loss = (
        torch.nn
        .BCEWithLogitsLoss()(
            logit,
            target
        )
    )

    assert torch.isfinite(
        logit
    ).all()

    assert torch.isfinite(
        loss
    ).all()

    loss.backward()

    assert model.kernels.grad is not None

    assert torch.isfinite(
        model.kernels.grad
    ).all()