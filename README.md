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

## How to use

The core of the project is ```bash the multi_controlled_u_circuit function, which takes the number of control qubits n and a 2x2 unitary matrix U as input.