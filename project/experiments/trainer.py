# %%
"""
trainer.py
Refactored training engine for Phase 2A Final Validation & Numerical Integrity Round.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import sys
import os
import csv
import json
import numpy as np
import pennylane as qml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.hybrid_classifier import HybridQConvClassifier
from data.mnist_preprocessing import get_mnist_datasets, extract_and_normalize_patches

def run_experiment(config):
    exp_name = config.get("exp_name", "experiment")
    init_mode = config.get("init_mode", "random")
    freeze_qconv = config.get("freeze_qconv", False)
    M = config.get("M", 2)
    R = S = config.get("R", 4)
    E = F = config.get("E", 4)
    epochs = config.get("epochs", 2)
    lr = config.get("lr", 0.01)
    seed = config.get("seed", 42)
    gradient_clipping = config.get("gradient_clipping", True)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    debug_numerics = config.get("debug_numerics", False)
    shots = config.get("shots", None)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*50}\nStarting Experiment: {exp_name} | Seed: {seed}\n{'='*50}")
    
    # 1. Setup Directories
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'phase2a', exp_name, f"seed_{seed}"))
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Data Loading
    train_set, val_set, test_set = get_mnist_datasets(seed=seed)
    # If the user specified fewer samples (e.g., diagnostic run)
    train_samples = config.get("train_samples", len(train_set))
    if train_samples < len(train_set):
        train_set = torch.utils.data.Subset(train_set, range(train_samples))
        
    train_loader = DataLoader(train_set, batch_size=config.get("batch_size", 1), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.get("batch_size", 1), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.get("batch_size", 1), shuffle=False)
    
    # We take a fixed test subset (100 samples) for feature map tracking
    eval_subset = [test_set[i] for i in range(min(100, len(test_set)))]
    
    # 3. Model Initialization
    model = HybridQConvClassifier(M=M, R=R, S=S, E=E, F=F, init_mode=init_mode)
    
    if freeze_qconv:
        model.kernels.requires_grad = False
    else:
        model.kernels.requires_grad = True
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Initial state capture
    initial_kernels_raw = model.kernels.detach().clone()
    initial_norms = torch.linalg.norm(initial_kernels_raw, dim=1, keepdim=True)
    initial_normalized = initial_kernels_raw / torch.where(initial_norms > 1e-9, initial_norms, torch.ones_like(initial_norms))
    
    # Extract feature maps before training
    model.eval()
    Y_init_list = []
    with torch.no_grad():
        for img, _ in eval_subset:
            _, y_map = model(img.unsqueeze(0), return_feature_map=True)
            Y_init_list.append(y_map.detach().clone())
    Y_init_tensor = torch.stack(Y_init_list)
    
    history = []
    start_time = time.time()
    
    # 4. Training Loop
    non_finite_events = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for step, (images, labels) in enumerate(train_loader):
            image = images[0]
            label = labels[0].float()
            
            optimizer.zero_grad()
            
            logit, _ = model(image.unsqueeze(0), return_feature_map=True)
            if debug_numerics and not torch.isfinite(logit).all():
                raise RuntimeError(f"Non-finite logit detected at epoch {epoch} step {step}")
                
            loss = criterion(logit, label)
            if debug_numerics and not torch.isfinite(loss).all():
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch} step {step}")
            
            loss.backward()
            
            # --- GRADIENT INSTRUMENTATION (Sections 2, 3, 4, 5) ---
            step_log = {"epoch": epoch, "step": step, "loss": loss.item()}
            
            non_finite_this_step = False
            
            if model.kernels.requires_grad:
                kernel_grad = model.kernels.grad
                if kernel_grad is not None:
                    grad_is_finite = torch.isfinite(kernel_grad).all().item()
                    
                    if not grad_is_finite:
                        non_finite_this_step = True
                        if not gradient_clipping:
                            raise RuntimeError("Non-finite gradient detected in parameter: kernels")
                        else:
                            # Sanitize to survive the StatePrep singularity!
                            model.kernels.grad = torch.nan_to_num(model.kernels.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                            
                    # Re-read after potential sanitize
                    grad_norm = torch.linalg.norm(model.kernels.grad).item()
                    grad_max = model.kernels.grad.abs().max().item()
                else:
                    grad_is_finite = False
                    grad_norm = float('nan')
                    grad_max = float('nan')
                    
                step_log["kernel_grad_finite"] = grad_is_finite
                step_log["kernel_grad_norm"] = grad_norm
                step_log["kernel_grad_max"] = grad_max
            
            # Check all parameters
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        non_finite_this_step = True
                        if not gradient_clipping:
                            raise RuntimeError(f"Non-finite gradient detected in parameter: {name}")
                        else:
                            parameter.grad = torch.nan_to_num(parameter.grad, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if non_finite_this_step:
                non_finite_events += 1
                
            if gradient_clipping:
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm).item()
                step_log["pre_clip_total_norm"] = pre_clip_norm
                if model.kernels.requires_grad and model.kernels.grad is not None:
                    step_log["post_clip_kernel_grad_norm"] = torch.linalg.norm(model.kernels.grad).item()
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = 1.0 if torch.sigmoid(logit).item() > 0.5 else 0.0
            train_preds.append(pred)
            train_targets.append(label.item())
            
            history.append(step_log)
            
            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch+1} | Step {step+1} | Loss: {loss.item():.4f}")
                
        # Epoch Validation
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                logit = model(images[0].unsqueeze(0))
                loss = criterion(logit, labels[0].float())
                val_loss += loss.item()
                pred = 1.0 if torch.sigmoid(logit).item() > 0.5 else 0.0
                val_preds.append(pred)
                val_targets.append(labels[0].float().item())
                
        train_acc = accuracy_score(train_targets, train_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss/len(val_loader):.4f} | Val Acc:   {val_acc:.4f}\n")
        
    total_time = time.time() - start_time
    
    # 5. Final Evaluation (Test Set)
    print("\n--- Final Test Evaluation ---")
    model.eval()
    test_loss, test_preds, test_targets = 0.0, [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logit = model(images[0].unsqueeze(0))
            loss = criterion(logit, labels[0].float())
            test_loss += loss.item()
            pred = 1.0 if torch.sigmoid(logit).item() > 0.5 else 0.0
            test_preds.append(pred)
            test_targets.append(labels[0].float().item())
            
    test_acc = accuracy_score(test_targets, test_preds)
    test_prec = precision_score(test_targets, test_preds, zero_division=0)
    test_rec = recall_score(test_targets, test_preds, zero_division=0)
    test_f1 = f1_score(test_targets, test_preds, zero_division=0)
    
    print(f"Test Loss: {test_loss/len(test_loader):.4f} | Test Acc: {test_acc:.4f}")
    
    # 6. Kernel & Feature Map Analysis
    final_kernels_raw = model.kernels.detach().clone()
    final_norms = torch.linalg.norm(final_kernels_raw, dim=1, keepdim=True)
    final_normalized = final_kernels_raw / torch.where(final_norms > 1e-9, final_norms, torch.ones_like(final_norms))
    
    raw_kernel_delta = torch.linalg.norm(final_kernels_raw - initial_kernels_raw).item()
    
    normalized_kernel_delta = {}
    kernel_cosine = {}
    for m in range(M):
        normalized_kernel_delta[f"m{m}"] = torch.linalg.norm(final_normalized[m] - initial_normalized[m]).item()
        kernel_cosine[f"m{m}"] = torch.dot(final_normalized[m], initial_normalized[m]).item()
        
    # Feature Map Movement
    Y_final_list = []
    with torch.no_grad():
        for img, _ in eval_subset:
            _, y_map = model(img.unsqueeze(0), return_feature_map=True)
            Y_final_list.append(y_map.detach().clone())
    Y_final_tensor = torch.stack(Y_final_list)
    
    feature_diff = Y_final_tensor - Y_init_tensor
    feature_mae = torch.mean(torch.abs(feature_diff)).item()
    feature_rmse = torch.sqrt(torch.mean(feature_diff**2)).item()
    feature_max_abs_difference = torch.max(torch.abs(feature_diff)).item()
    relative_feature_change = (torch.linalg.norm(feature_diff) / (torch.linalg.norm(Y_init_tensor) + 1e-9)).item()
    
    # Classical vs Quantum Correctness
    # Y_classical = K^T X
    c_vs_q_mae_list = []
    c_vs_q_rmse_list = []
    c_vs_q_max_list = []
    with torch.no_grad():
        for idx, (img, _) in enumerate(eval_subset):
            # manual classical patch extraction
            # X_p shape [E*F, R*S]
            patches = []
            img_sq = img.squeeze(0) # [7, 7]
            for i in range(E):
                for j in range(F):
                    patch = img_sq[i:i+R, j:j+S].flatten()
                    patches.append(patch)
            patches = torch.stack(patches) # [E*F, R*S]
            
            # Classical convolution: Y = K @ X^T -> [M, R*S] @ [R*S, E*F] -> [M, E*F]
            y_classical = torch.matmul(final_kernels_raw, patches.T) # [M, E*F]
            y_classical = y_classical.view(M, E, F)
            
            y_quantum = Y_final_list[idx]
            diff = y_quantum - y_classical
            c_vs_q_mae_list.append(torch.mean(torch.abs(diff)).item())
            c_vs_q_rmse_list.append(torch.sqrt(torch.mean(diff**2)).item())
            c_vs_q_max_list.append(torch.max(torch.abs(diff)).item())
            
    c_vs_q_mae = np.mean(c_vs_q_mae_list)
    c_vs_q_rmse = np.mean(c_vs_q_rmse_list)
    c_vs_q_max = np.max(c_vs_q_max_list)
    
    # 7. Write Results
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "test_acc": test_acc, "test_f1": test_f1, "test_loss": test_loss/len(test_loader),
            "raw_kernel_delta": raw_kernel_delta,
            "normalized_kernel_delta": normalized_kernel_delta,
            "kernel_cosine": kernel_cosine,
            "initial_kernel_norms": initial_norms.flatten().tolist(),
            "final_kernel_norms": final_norms.flatten().tolist(),
            "non_finite_gradient_events": non_finite_events,
            "feature_mae": feature_mae,
            "feature_rmse": feature_rmse,
            "feature_max_abs_difference": feature_max_abs_difference,
            "relative_feature_change": relative_feature_change,
            "c_vs_q_mae": c_vs_q_mae,
            "c_vs_q_rmse": c_vs_q_rmse,
            "c_vs_q_max": c_vs_q_max,
            "runtime": total_time
        }, f, indent=4)
        
    with open(os.path.join(out_dir, "step_logs.csv"), "w", newline="") as f:
        if history:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
            
    print("\n--- Correctness Check ---")
    print(f"Classical vs Quantum MAE:  {c_vs_q_mae:.8f}")
    print(f"Classical vs Quantum RMSE: {c_vs_q_rmse:.8f}")
    print(f"Classical vs Quantum MAX:  {c_vs_q_max:.8f}")
    
    print("\n--- Kernel Movement ---")
    print(f"Raw Delta: {raw_kernel_delta:.4f}")
    print(f"Norm Delta: {normalized_kernel_delta}")
    
    print("\n--- Feature Map Movement ---")
    print(f"Feature MAE: {feature_mae:.4f}")
    print(f"Relative Feature Change: {relative_feature_change:.4f}")
    
    # 8. Shot Robustness Evaluation
    shot_results = {}
    if not config.get('skip_shots', False):
        print('\n--- Shot Robustness Evaluation ---')
        for shots in [1024, 4096, 8192]:
            model.set_shots(shots)
            
            s_mae_list, s_rmse_list, s_max_list = [], [], []
            s_preds, s_targets = [], []
            
            with torch.no_grad():
                for idx, (img, label) in enumerate(eval_subset):
                    logit, y_quantum_shot = model(img.unsqueeze(0), return_feature_map=True)
                    pred = 1.0 if torch.sigmoid(logit).item() > 0.5 else 0.0
                    s_preds.append(pred)
                    s_targets.append(float(label))
                    
                    # Compare to analytic Y_final
                    y_analytic = Y_final_list[idx]
                    diff = y_quantum_shot - y_analytic
                    s_mae_list.append(torch.mean(torch.abs(diff)).item())
                    s_rmse_list.append(torch.sqrt(torch.mean(diff**2)).item())
                    s_max_list.append(torch.max(torch.abs(diff)).item())
                    
            shot_acc = accuracy_score(s_targets, s_preds)
            shot_f1 = f1_score(s_targets, s_preds, zero_division=0)
            
            shot_results[f'shots_{shots}'] = {
                'mae': np.mean(s_mae_list),
                'rmse': np.mean(s_rmse_list),
                'max_error': np.max(s_max_list),
                'acc': shot_acc,
                'f1': shot_f1
            }
            print(f'Shots: {shots} | Acc: {shot_acc:.4f} | MAE: {np.mean(s_mae_list):.4f}')
            
        # Write shot results
        with open(os.path.join(out_dir, 'shot_robustness.json'), 'w') as f:
            json.dump(shot_results, f, indent=4)
            
    # 9. Resource Analysis
    print('\n--- Resource Analysis ---')
    model.set_shots(None)
    try:
        dummy_patches = torch.zeros(E*F, R*S)
        dummy_patches[0, 0] = 1.0 # Give it a valid norm
        specs = qml.specs(model.qnode)(final_normalized, dummy_patches)
        with open(os.path.join(out_dir, 'resource_summary.json'), 'w') as f:
            safe_specs = {k: str(v) if not isinstance(v, (int, float, str, bool)) else v for k, v in specs.items()}
            json.dump(safe_specs, f, indent=4)
        print('Total Qubits:', specs.get('num_device_wires', 'unknown'))
    except Exception as e:
        print('Resource analysis skipped:', e)
    
    return True
