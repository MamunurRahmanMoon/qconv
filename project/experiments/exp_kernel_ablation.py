# %%
"""
exp_kernel_ablation.py
Ablation (Part D4): Seed initialized Trainable Kernels.
Trains BOTH the Quantum Convolution kernels and the Linear classifier, starting from the seed edge detectors.
"""

# %%
from trainer import run_experiment

# %%
if __name__ == "__main__":
    run_experiment(
        exp_name="Ablation - Seed Trainable Kernels",
        init_mode="seed",
        freeze_qconv=False,
        M=2,
        epochs=2,
        lr=0.01,
        seed=42
    )
