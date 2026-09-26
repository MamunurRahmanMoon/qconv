# %%
"""
qconv_reconstruction.py
Handles the classical post-processing of the quantum measurement probabilities.
"""

# %%
import torch
import math

# %%
def reconstruct_from_probs_torch(probs, kernel_norms, patch_norms, M, E, F):
    """
    Reconstructs the feature map using the raw probabilities and the raw norms.
    """
    num_spatial_qubits = math.ceil(math.log2(E * F)) if E * F > 1 else 1
    num_filter_qubits = math.ceil(math.log2(M)) if M > 1 else 1
    
    probs_reshaped = probs.reshape(2, 2**num_spatial_qubits, 2**num_filter_qubits)
    
    p0 = probs_reshaped[0, :E*F, :M]
    p1 = probs_reshaped[1, :E*F, :M]
    
    total_prob = p0 + p1
    
    # Safe division to prevent gradient explosion (d/dx (1/1e-12) -> infinity)
    safe_total = torch.where(total_prob > 1e-9, total_prob, torch.ones_like(total_prob))
    p0_cond = torch.where(total_prob > 1e-9, p0 / safe_total, torch.zeros_like(p0))
    p1_cond = torch.where(total_prob > 1e-9, p1 / safe_total, torch.zeros_like(p1))
    
    estimated_inner_product = p0_cond - p1_cond
    
    # Re-apply the raw norms! (Part B2 Requirement)
    rescaled = estimated_inner_product * patch_norms.unsqueeze(1) * kernel_norms.unsqueeze(0)
    
    Y_reconstructed = rescaled.view(E, F, M).permute(2, 0, 1)
    return Y_reconstructed
