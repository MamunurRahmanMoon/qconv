# %%
"""
exp_smoke_test.py

Minimal end-to-end Phase-2A sanity test.

Runs:
    1 training sample
    1 epoch
    2 validation samples
    2 test samples

The purpose is to verify that the whole pipeline works before
launching long Colab experiments.
"""

from trainer import run_experiment


if __name__ == "__main__":

    config = {

        "exp_name":
            "Smoke Test",

        "init_mode":
            "random",

        "freeze_qconv":
            False,

        "M":
            2,

        "R":
            4,

        "S":
            4,

        "E":
            4,

        "F":
            4,

        "epochs":
            1,

        "lr":
            0.01,

        "seed":
            42,

        "batch_size":
            1,

        "gradient_clipping":
            True,

        "max_grad_norm":
            1.0,

        "debug_numerics":
            True,

        "fail_on_nonfinite_grad":
            True,

        "train_samples":
            1,

        "val_samples":
            2,

        "test_samples":
            2,

        "analyze_features":
            True,

        "feature_eval_samples":
            1,

        "run_shot_eval":
            False,

        "run_resource_analysis":
            False,
    }

    run_experiment(
        config
    )

    print(
        "\n"
        + "=" * 60
        + "\nSMOKE TEST PASSED"
        + "\n"
        + "=" * 60
    )