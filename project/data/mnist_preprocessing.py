# %%
"""
mnist_preprocessing.py
Handles extraction and mathematically strict normalization of image patches.
"""

# %%
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# %%
def extract_and_normalize_patches(image_tensor, R, S):
    """
    Extracts R*S patches. 
    Implements the explicit Zero-Patch Policy (Part B1).
    """
    image = image_tensor.squeeze()
    H, W = image.shape
    E, F = H - R + 1, W - S + 1
    
    patches = []
    patch_norms = []
    
    # Safe basis state for zero patches (e.g. |0...0>)
    zero_safe_state = torch.zeros(R * S, dtype=torch.float32, device=image_tensor.device)
    zero_safe_state[0] = 1.0
    
    for i in range(E):
        for j in range(F):
            patch = image[i:i+R, j:j+S].flatten()
            norm = torch.linalg.norm(patch)
            
            if norm == 0.0:
                normalized_patch = zero_safe_state
                recorded_norm = torch.tensor(0.0, device=image_tensor.device)
            else:
                normalized_patch = patch / norm
                recorded_norm = norm
                
            patches.append(normalized_patch)
            patch_norms.append(recorded_norm)
            
    return torch.stack(patches), torch.stack(patch_norms), E, F

# %%
def get_mnist_datasets(seed=42):
    """
    Downloads and prepares deterministic splits for MNIST 0 vs 1 (Part C).
    Train: 1000, Val: 200, Test: 500
    """
    # Deterministic behavior
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    transform = transforms.Compose([
        transforms.Resize((7, 7)),
        transforms.ToTensor()
    ])
    
    full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    full_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    def filter_and_balance(dataset, num_per_class):
        idx_0 = torch.where(dataset.targets == 0)[0]
        idx_1 = torch.where(dataset.targets == 1)[0]
        
        # Shuffle indices
        idx_0 = idx_0[torch.randperm(len(idx_0))]
        idx_1 = idx_1[torch.randperm(len(idx_1))]
        
        # Take num_per_class from each
        selected_idx_0 = idx_0[:num_per_class]
        selected_idx_1 = idx_1[:num_per_class]
        
        selected_idx = torch.cat([selected_idx_0, selected_idx_1])
        selected_idx = selected_idx[torch.randperm(len(selected_idx))] # Shuffle combined
        
        dataset.targets = dataset.targets[selected_idx]
        dataset.data = dataset.data[selected_idx]
        return dataset
        
    # We want Train (1000) -> 500 per class
    # We want Val (200) -> 100 per class
    # So we extract 600 per class from the training set, and split it.
    
    # 1. Prepare Train + Val pool
    idx_train_0 = torch.where(full_train.targets == 0)[0][torch.randperm(len(torch.where(full_train.targets == 0)[0]))]
    idx_train_1 = torch.where(full_train.targets == 1)[0][torch.randperm(len(torch.where(full_train.targets == 1)[0]))]
    
    train_idx = torch.cat([idx_train_0[:500], idx_train_1[:500]])
    val_idx = torch.cat([idx_train_0[500:600], idx_train_1[500:600]])
    
    train_subset = torch.utils.data.Subset(full_train, train_idx[torch.randperm(1000)])
    val_subset = torch.utils.data.Subset(full_train, val_idx[torch.randperm(200)])
    
    # 2. Prepare Test pool (500 -> 250 per class)
    idx_test_0 = torch.where(full_test.targets == 0)[0][torch.randperm(len(torch.where(full_test.targets == 0)[0]))]
    idx_test_1 = torch.where(full_test.targets == 1)[0][torch.randperm(len(torch.where(full_test.targets == 1)[0]))]
    test_idx = torch.cat([idx_test_0[:250], idx_test_1[:250]])
    test_subset = torch.utils.data.Subset(full_test, test_idx[torch.randperm(500)])
    
    print(f"Data Splits Balanced:")
    print(f"Train: {len(train_subset)} samples (500 zeros, 500 ones)")
    print(f"Val: {len(val_subset)} samples (100 zeros, 100 ones)")
    print(f"Test: {len(test_subset)} samples (250 zeros, 250 ones)")
    
    return train_subset, val_subset, test_subset
