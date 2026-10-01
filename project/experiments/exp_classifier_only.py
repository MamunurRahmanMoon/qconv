# %%
"""
exp_classifier_only.py

Baseline 0:
    Randomly initialized QConv kernels are frozen.
    Only the classical classifier is trained.

Runs five random seeds.
"""

from trainer import run_experiment


if __name__ == "__main__":

    seeds = [
        0,
        1,
        2,
        3,
        4,
    ]

    for seed in seeds:

        config = {

            "exp_name":
                "Baseline 0 - Random Fixed Kernels",

            "init_mode":
                "random",

            "freeze_qconv":
                True,

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
                2,

            "lr":
                0.01,

            "seed":
                seed,

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
                1000,

            "val_samples":
                200,

            "test_samples":
                500,

            "analyze_features":
                True,

            "feature_eval_samples":
                20,

            # IMPORTANT:
            # These are intentionally disabled here.
            "run_shot_eval":
                False,

            "run_resource_analysis":
                False,
        }

        run_experiment(
            config
        )