# %%
"""
hybrid_classifier.py
The PyTorch nn.Module bridging the classical layers and the QNode.
"""

# %%
import torch
import torch.nn as nn
from qconv.qconv_quantum import create_qconv_qnode
from qconv.qconv_reconstruction import reconstruct_from_probs_torch
from data.mnist_preprocessing import extract_and_normalize_patches

# %%
class HybridQConvClassifier(nn.Module):
    def __init__(self, M, R, S, E, F, init_mode="random"):
        super(HybridQConvClassifier, self).__init__()
        self.M = M
        self.R = R
        self.S = S
        self.E = E
        self.F = F
        self.init_mode = init_mode
        
        # Raw kernel parameters
        if init_mode == "random":
            initial_weights = torch.rand(M, R * S) + 1e-3
        elif init_mode == "seed":
            # Using the seed paper's fixed classical edge detectors
            # Resized/flattened to match R*S length, assuming R=4, S=4
            assert R*S == 16, "Seed init currently hardcoded for 4x4 kernels"
            # We construct simple vertical and horizontal edge filters for the sake of the baseline
            k1 = torch.tensor([[1,1,-1,-1],[1,1,-1,-1],[1,1,-1,-1],[1,1,-1,-1]], dtype=torch.float32).flatten()
            k2 = torch.tensor([[1,1,1,1],[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1]], dtype=torch.float32).flatten()
            initial_weights = torch.stack([k1, k2])
        else:
            raise ValueError("Unknown init_mode")
            
        self.kernels = nn.Parameter(initial_weights)
        
        # QNode initialization
        self.qnode = create_qconv_qnode(R, S, M, E, F)
        
        # Classical classifier
        self.fc = nn.Linear(M * E * F, 1)

    def set_shots(self, shots):
        """Swaps the QNode for finite shot evaluation."""
        self.qnode = create_qconv_qnode(self.R, self.S, self.M, self.E, self.F, shots=shots)

    def forward(self, x, return_feature_map=False):
        normalized_patches, patch_norms, _, _ = extract_and_normalize_patches(x, self.R, self.S)
        
        # Apply strict kernel normalization policy (Part B2)
        kernel_norms = torch.linalg.norm(self.kernels, dim=1)
        
        # Handle zero-kernels just like zero-patches for mathematical safety
        # If a kernel is 0, we MUST pass a valid norm-1 state to PennyLane (e.g. |0...0>)
        normalized_kernels = []
        for i in range(self.M):
            if kernel_norms[i] < 1e-9:
                safe_state = torch.zeros_like(self.kernels[i])
                safe_state[0] = 1.0
                # Detach so we don't backprop through the dummy state replacement
                normalized_kernels.append(safe_state.detach())
            else:
                normalized_kernels.append(self.kernels[i] / kernel_norms[i])
        normalized_kernels = torch.stack(normalized_kernels)
        
        # Quantum evaluation
        probs = self.qnode(normalized_kernels, normalized_patches)
        
        # Reconstruct exactly as Y = <K,X> * ||K|| * ||X||
        feature_map = reconstruct_from_probs_torch(probs, kernel_norms, patch_norms, self.M, self.E, self.F)
        
        # Flatten and classify
        flattened = feature_map.flatten().float()
        logit = self.fc(flattened)
        
        # Squeeze out the extra dimension to match target shape [batch] if needed
        out_logit = logit.squeeze()
        
        if return_feature_map:
            return out_logit, feature_map
        return out_logit
