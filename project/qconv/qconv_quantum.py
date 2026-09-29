# %%
"""
qconv_quantum.py
Defines the QNode for the Hadamard Test.
"""

# %%
import pennylane as qml
import torch
import numpy as np
from qconv.qconv_oracles import apply_uk_oracle_trainable, apply_ux_oracle_trainable

# %%
def create_qconv_qnode(R, S, M, E, F, shots=None):
    """
    Dynamically creates and returns a PennyLane QNode configured for backprop (or finite shots).
    """
    num_data_qubits = int(np.ceil(np.log2(R * S)))
    num_spatial_qubits = int(np.ceil(np.log2(E * F))) if E * F > 1 else 1
    num_filter_qubits = int(np.ceil(np.log2(M))) if M > 1 else 1
    
    ancilla_wire = 0
    spatial_wires = list(range(1, 1 + num_spatial_qubits))
    filter_wires = list(range(1 + num_spatial_qubits, 1 + num_spatial_qubits + num_filter_qubits))
    data_wires = list(range(1 + num_spatial_qubits + num_filter_qubits, 1 + num_spatial_qubits + num_filter_qubits + num_data_qubits))
    total_wires = 1 + num_spatial_qubits + num_filter_qubits + num_data_qubits
    
    dev = qml.device('default.qubit', wires=total_wires, shots=shots)
    
    diff_method = "backprop" if shots is None else "best"
    
    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def qconv_hadamard_test(normalized_kernels, normalized_patches):
        for w in [ancilla_wire] + spatial_wires + filter_wires:
            qml.Hadamard(wires=w)
            
        def uk_wrapper():
            apply_uk_oracle_trainable(normalized_kernels, M, filter_wires, data_wires)
        qml.ctrl(uk_wrapper, control=ancilla_wire, control_values=[0])()
        
        def ux_wrapper():
            apply_ux_oracle_trainable(normalized_patches, E * F, spatial_wires, data_wires)
        qml.ctrl(ux_wrapper, control=ancilla_wire, control_values=[1])()
        
        qml.Hadamard(wires=ancilla_wire)
        return qml.probs(wires=[ancilla_wire] + spatial_wires + filter_wires)
        
    return qconv_hadamard_test
