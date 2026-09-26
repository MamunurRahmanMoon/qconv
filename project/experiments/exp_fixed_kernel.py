# %%
"""
exp_fixed_kernel.py
Baseline 1 (Part D2): Seed Fixed Kernels.
Freezes the manually handcrafted seed edge-detector kernels and trains only the Linear classifier.
"""

# %%
from trainer import run_experiment

# %%
if __name__ == "__main__":
    run_experiment(
        exp_name="Baseline 1 - Fixed Seed Kernels",
        init_mode="seed",
        freeze_qconv=True,
        M=2,
        epochs=2,
        lr=0.01,
        seed=42
    )
