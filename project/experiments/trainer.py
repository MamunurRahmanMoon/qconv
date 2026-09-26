# %%
"""
trainer.py
Generic training engine that supports freezing/unfreezing layers, 
tracking Train/Val/Test metrics, calculating F1, and analyzing kernel movement.
"""

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import sys
import os

# Ensure the project root is in the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.hybrid_classifier import HybridQConvClassifier
from data.mnist_preprocessing import get_mnist_datasets

# %%
def run_experiment(exp_name, init_mode, freeze_qconv, M=2, epochs=2, lr=0.01, seed=42):
    print(f"==================================================")
    print(f"Running Experiment: {exp_name}")
    print(f"init_mode: {init_mode}, freeze_qconv: {freeze_qconv}, M: {M}, seed: {seed}")
    print(f"==================================================")
    
    # 1. Load Data
    train_set, val_set, test_set = get_mnist_datasets(seed=seed)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    # 2. Initialize Model
    # R=4, S=4, E=4, F=4 for a 7x7 image with 4x4 patches
    model = HybridQConvClassifier(M=M, R=4, S=4, E=4, F=4, init_mode=init_mode)
    
    if freeze_qconv:
        model.kernels.requires_grad = False
    else:
        model.kernels.requires_grad = True
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Track initial kernels
    initial_kernels = model.kernels.detach().clone()
    
    # 3. Training Loop
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for i, (images, labels) in enumerate(train_loader):
            # For 7x7 we just use the first image in the batch (batch_size=1)
            image = images[0]
            label = labels[0].float()
            
            optimizer.zero_grad()
            logit = model(image)
            loss = criterion(logit, label)
            
            loss.backward()
            
            # Clip gradients to prevent PennyLane's MottonenStatePreparation from exploding near amplitude 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Print gradients on first step of first epoch
            if epoch == 0 and i == 0:
                print("\n[Gradient Check Step 0]")
                if not freeze_qconv:
                    print(f"Kernel Grad Norm: {model.kernels.grad.norm().item():.6f}")
                print(f"Linear Grad Norm: {model.fc.weight.grad.norm().item():.6f}\n")
                
            optimizer.step()
            
            train_loss += loss.item()
            pred = 1.0 if torch.sigmoid(logit).item() > 0.5 else 0.0
            train_preds.append(pred)
            train_targets.append(label.item())
            
            # Print occasionally so we know it's running
            if (i+1) % 200 == 0:
                print(f"Epoch {epoch+1} | Step {i+1}/1000 | Loss: {loss.item():.4f}")
                
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                image = images[0]
                label = labels[0].float()
                logit = model(image)
                loss = criterion(logit, label)
                val_loss += loss.item()
                pred = 1.0 if torch.sigmoid(logit).item() > 0.5 else 0.0
                val_preds.append(pred)
                val_targets.append(label.item())
                
        train_acc = accuracy_score(train_targets, train_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss/len(val_loader):.4f} | Val Acc:   {val_acc:.4f}\n")
        
    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")
    
    # 4. Final Evaluation (Test Set)
    print("\n--- Final Test Evaluation ---")
    model.eval()
    test_loss = 0.0
    test_preds, test_targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            image = images[0]
            label = labels[0].float()
            logit = model(image)
            loss = criterion(logit, label)
            test_loss += loss.item()
            pred = 1.0 if torch.sigmoid(logit).item() > 0.5 else 0.0
            test_preds.append(pred)
            test_targets.append(label.item())
            
    test_acc = accuracy_score(test_targets, test_preds)
    test_prec = precision_score(test_targets, test_preds, zero_division=0)
    test_rec = recall_score(test_targets, test_preds, zero_division=0)
    test_f1 = f1_score(test_targets, test_preds, zero_division=0)
    
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"Test Acc:  {test_acc:.4f}")
    print(f"Test Prec: {test_prec:.4f}")
    print(f"Test Rec:  {test_rec:.4f}")
    print(f"Test F1:   {test_f1:.4f}")
    
    # 5. Kernel Delta Verification
    final_kernels = model.kernels.detach()
    delta_K = torch.linalg.norm(final_kernels - initial_kernels).item()
    print(f"\nKernel Update Delta (||K_final - K_initial||): {delta_K:.6f}")
    
    return {
        "test_acc": test_acc,
        "test_f1": test_f1,
        "delta_K": delta_K
    }
