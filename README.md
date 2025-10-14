Multi-Controlled U Gate Implementation in Qiskit
This repository contains a Python implementation of a multi-controlled unitary gate (C 
n
 U) using Qiskit. This type of gate is a fundamental primitive in many advanced quantum algorithms, including Shor's algorithm and Grover's search.

The implemented solution is based on the recursive decomposition method described by Barenco et al. in their 1995 paper, "Elementary Gates for Quantum Computation."

Overview
A multi-controlled unitary gate, C 
n
 U, is a quantum gate that applies a single-qubit unitary operation U to a target qubit if and only if all n control qubits are in the state ∣1⟩. If any control qubit is in the state ∣0⟩, the gate acts as the identity.

This project provides a Qiskit function that takes an integer n (the number of controls) and a 2×2 unitary matrix U and returns a QuantumCircuit object that correctly implements the C 
n
 U operation.

Methodology: Recursive Decomposition
The core of the implementation is a recursive algorithm that constructs a C 
n
 U gate from gates with fewer controls. The key insight is that a C 
n
 U gate can be built from singly-controlled V gates (where V 
2
 =U) and (n−1)-controlled gates.

The general recursive step for n>1 is a five-step sequence:

Apply a singly-controlled V gate on the target, controlled by the last control qubit.

Apply a C 
n−1
 X (multi-controlled Toffoli) gate, targeting the last control qubit.

Apply a singly-controlled V 
†
  gate on the target, controlled by the last control qubit.

Apply the same C 
n−1
 X gate again.

Apply a C 
n−1
 V gate (the recursive call) on the target, controlled by the first n−1 qubits.

This sequence cleverly ensures that the full U operation is synthesized only when all n controls are active.

Installation
The code requires the following Python libraries:

pip install qiskit numpy scipy

Usage
The main function is get_multi_controlled_u_circuit_corrected, located in the notebook.

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator
import numpy as np

# (Assuming the function get_multi_controlled_u_circuit_corrected is defined)

# --- Example: Create a CCX (Toffoli) gate ---
# This is a C^2(X) gate, so n=2 and U is the Pauli-X matrix.

n_controls = 2
X_gate = np.array([[0, 1], [1, 0]])

# Build the circuit
ccx_circuit = get_multi_controlled_u_circuit_corrected(n_controls, X_gate)

print("--- Generated CCX Circuit ---")
print(ccx_circuit)

# --- Verification ---
# Compare the generated circuit's unitary to Qiskit's native mcx gate
reference_qc = QuantumCircuit(n_controls + 1)
reference_qc.mcx(list(range(n_controls)), n_controls)

# Check if the operators are equivalent (equal up to a global phase)
op_generated = Operator(ccx_circuit)
op_reference = Operator(reference_qc)

print(f"\nIs the generated circuit equivalent to Qiskit's native MCX? {op_generated.equiv(op_reference)}")

Resource Complexity Analysis
The resource requirements for this decomposition scale with the number of control qubits, n.

Resource

Asymptotic Complexity

Gate Count

O(n 
2
 )

Circuit Depth

O(n 
2
 )

Ancilla Qubits

O(n)

The analysis shows that while this construction is general and powerful, it is resource-intensive, with gate count and depth growing quadratically. This highlights the significant overhead associated with implementing complex logical operations on quantum computers.
