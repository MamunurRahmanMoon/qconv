# %%
"""
qconv_oracles.py
Contains the quantum oracles U_K (Kernel) and U_X (Image Patches).
"""

# %%
import pennylane as qml
import torch

# %%
def apply_uk_oracle_trainable(normalized_kernels, M, filter_wires, data_wires):
    """
    Applies the U_K oracle using parameterized/trainable classical tensors.
    """
    for m in range(M):
        state = normalized_kernels[m]
        ctrl_values = [int(x) for x in format(m, f'0{len(filter_wires)}b')]
        qml.ctrl(qml.StatePrep, control=filter_wires, control_values=ctrl_values)(state, wires=data_wires)

# %%
def apply_ux_oracle_trainable(normalized_patches, num_patches, spatial_wires, data_wires):
    """
    Applies the U_X oracle to load image patch states.
    """
    for p_spatial in range(num_patches):
        state = normalized_patches[p_spatial]
        ctrl_values = [int(x) for x in format(p_spatial, f'0{len(spatial_wires)}b')]
        qml.ctrl(qml.StatePrep, control=spatial_wires, control_values=ctrl_values)(state, wires=data_wires)
