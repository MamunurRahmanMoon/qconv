import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.hybrid_classifier import HybridQConvClassifier

def test_kernel_zero():
    model = HybridQConvClassifier(1, 4, 4, 4, 4, "random")
    
    # 1. Force kernel to zero
    with torch.no_grad():
        model.kernels.zero_()
        
    img = torch.rand(1, 7, 7)
    
    # 2 & 4. Verify no NaN occurs in forward pass (handles safe state internally)
    try:
        logit, feature_map = model(img.unsqueeze(0), return_feature_map=True)
    except Exception as e:
        assert False, f"Forward pass failed with exception: {e}"
        
    # 3. Verify reconstructed contribution is zero
    assert torch.all(feature_map == 0.0), f"Feature map should be 0.0 for zero kernels, got {feature_map}"
    
    print("[PASS] test_kernel_zero")

if __name__ == "__main__":
    test_kernel_zero()
