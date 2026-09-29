import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from data.mnist_preprocessing import extract_and_normalize_patches
from qconv.qconv_quantum import create_qconv_qnode
from qconv.qconv_reconstruction import reconstruct_from_probs_torch

def test_zero_patch():
    # Construct a 4x4 zero patch
    img = torch.zeros(1, 4, 4)
    normalized_patches, patch_norms, E, F = extract_and_normalize_patches(img, 4, 4)
    
    # 1. Verify patch_norm == 0
    assert torch.all(patch_norms == 0.0), f"Patch norms should be 0, got {patch_norms}"
    
    # 2. Verify state passed to StatePrep is valid (norm 1.0)
    norm_val = torch.linalg.norm(normalized_patches, dim=1)
    assert torch.allclose(norm_val, torch.tensor([1.0])), f"Safe state norm is not 1.0, got {norm_val}"
    
    # 3. Create dummy kernel
    kernels = torch.rand(1, 16)
    kernel_norms = torch.linalg.norm(kernels, dim=1)
    normalized_kernels = kernels / kernel_norms.unsqueeze(1)
    
    # Evaluate quantum circuit
    qnode = create_qconv_qnode(4, 4, 1, E, F)
    probs = qnode(normalized_kernels, normalized_patches)
    
    # 4. Verify reconstructed output == 0
    reconstructed = reconstruct_from_probs_torch(probs, kernel_norms, patch_norms, 1, E, F)
    assert torch.all(reconstructed == 0.0), f"Reconstructed output should be 0, got {reconstructed}"
    
    print("[PASS] test_zero_patch")

if __name__ == "__main__":
    test_zero_patch()
