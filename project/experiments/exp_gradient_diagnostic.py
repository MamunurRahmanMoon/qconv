# %%
"""
exp_gradient_diagnostic.py
No-clipping diagnostic experiment (Section 6).
Determine whether the previous failure (NaN gradients) can be reproduced without clipping.
"""

from trainer import run_experiment
import json

if __name__ == "__main__":
    config = {
        "exp_name": "gradient_diagnostic",
        "init_mode": "random",
        "freeze_qconv": False,
        "M": 2,
        "R": 4,
        "E": 4,
        "epochs": 1,
        "lr": 0.01,
        "seed": 42,
        "train_samples": 100, # diagnostic subset
        "batch_size": 1,
        "gradient_clipping": False, # DO NOT HIDE THE PROBLEM
        "debug_numerics": True
    }
    
    try:
        run_experiment(config)
        print("\n[DIAGNOSTIC] Training completed successfully without non-finite gradients.")
    except Exception as e:
        print(f"\n[DIAGNOSTIC] Training failed with exception:\n{e}")
