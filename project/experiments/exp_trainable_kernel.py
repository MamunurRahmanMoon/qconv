# %%
"""
exp_trainable_kernel.py
Main Phase 2A Model (Part D3): Random initialized Trainable Kernels.
Trains BOTH the Quantum Convolution kernels and the Linear classifier.
"""

# %%
from trainer import run_experiment

# %%
if __name__ == "__main__":
    run_experiment(
        exp_name="Main Model - Random Trainable Kernels",
        init_mode="random",
        freeze_qconv=False,
        M=2,
        epochs=2,
        lr=0.01,
        seed=42
    )
