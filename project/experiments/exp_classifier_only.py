# %%
"""
exp_classifier_only.py
Baseline 0 (Part D1): Random Fixed Kernels.
Freezes completely random kernels and trains only the Linear classifier.
Answers: How much does the classifier alone contribute?
"""

# %%
from trainer import run_experiment

# %%
if __name__ == "__main__":
    run_experiment(
        exp_name="Baseline 0 - Random Fixed Kernels (Classifier Only)",
        init_mode="random",
        freeze_qconv=True,
        M=2,
        epochs=2,
        lr=0.01,
        seed=42
    )
