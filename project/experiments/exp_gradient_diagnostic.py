# %%
"""
exp_gradient_diagnostic.py

Purpose:
    Determine whether the historical non-finite gradient problem
    still occurs when gradient clipping is completely disabled.

This is NOT a full training experiment.
"""

from trainer import run_experiment


if __name__ == "__main__":

    config = {

        "exp_name":
            "Gradient Diagnostic",

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

        # IMPORTANT:
        # We intentionally disable clipping here.
        "gradient_clipping":
            False,

        "max_grad_norm":
            1.0,

        "debug_numerics":
            True,

        "fail_on_nonfinite_grad":
            True,

        # Tiny diagnostic run.
        "train_samples":
            20,

        "val_samples":
            20,

        "test_samples":
            20,

        # Do not waste time on large post-analysis.
        "analyze_features":
            False,

        "feature_eval_samples":
            5,

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
        + "\nGRADIENT DIAGNOSTIC PASSED"
        + "\nNo non-finite gradient was observed."
        + "\n"
        + "=" * 60
    )