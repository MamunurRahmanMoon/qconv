# %%
from trainer import run_experiment

if __name__ == "__main__":
    for seed in [0, 1, 2, 3, 4]:
        run_experiment({
            "exp_name": "Ablation - Seed Trainable",
            "init_mode": "seed",
            "freeze_qconv": False,
            "M": 2, "R": 4, "E": 4,
            "epochs": 2, "lr": 0.01,
            "seed": seed, "gradient_clipping": True
        })
