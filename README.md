# Multi-Controlled U Gate Implementation in Qiskit

This project provides a Python function in Qiskit to construct a multi-controlled unitary gate, denoted as $C^n U$. Such gates are fundamental in many advanced quantum algorithms, like Shor's and Grover's algorithms.

The implementation is based on the recursive decomposition method described by Barenco et al. (1995), which builds a $C^n U$ gate from lower-order controlled gates.

## Features
* **Recursive Decomposition**: Implements the $C^n U$ gate using a recursive approach.
* **Qiskit Integration**: Seamlessly integrates with the Qiskit quantum computing framework.
* **Verification**: The provided notebook includes a verification function to ensure the constructed circuit behaves as expected.
* **Circuit Visualization**: You can save a diagram of the generated quantum circuit.

## Dependencies
To use this project, you will need the following libraries:
* numpy
* qiskit
* scipy

You can install them using pip:
```bash
pip install numpy qiskit scipy
```

## How to use

The core of the project is the ```bash multi_controlled_u_circuit``` function, which takes the number of control qubits n and a 2x2 unitary matrix U as input. Here is a simple example of how to build a $C^3 U$ gate:

```bash
import numpy as np
from qiskit.circuit import QuantumCircuit
from controlled_u_gate import multi_controlled_u_circuit # Assuming the function is in this file
```

* 1. Define the unitary matrix U
```bash
theta = np.pi / 4
U_matrix = np.array([
    [np.cos(theta/2), -1j * np.sin(theta/2)],
    [-1j * np.sin(theta/2), np.cos(theta/2)]
])
```
* 2. Set the number of control qubits
```bash
n_controls = 3
```

* 3. Build the circuit
```bash
circuit = multi_controlled_u_circuit(n_controls, U_matrix)
```
* 4. Print and draw the circuit
```bash
print(circuit)
circuit.draw('mpl')
```

The run function in the controlled-u-gate.ipynb notebook provides a more detailed example, including a verification step.

## Complexity Analysis

The resource complexity of this implementation is as follows:
* Gate Count: $O (n^2)$
* Circuit Depth: $O (n^2)$
* Ancilla Qubits:  $O (n)$