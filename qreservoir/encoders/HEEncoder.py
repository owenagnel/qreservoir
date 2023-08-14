from qulacs import QuantumCircuit, QuantumState
import numpy as np
from numpy.typing import NDArray
from qulacsvis import circuit_drawer
from qulacs.gate import CZ, RotX
from qreservoir.encoders.Encoder import Encoder


class HEEncoder(Encoder):
    """Simple Hardware Effecient Encoder class. Implements a
    reuploading stretegy using CZ and RX gates with a cyclic
    entangling structure"""

    def __init__(self, input_size: int, depth: int) -> None:
        self.input_size = input_size
        self.depth = depth

    def get_circuit(self, input_vect: NDArray[np.double]) -> QuantumCircuit:
        """Generates the parameterised circuit for the given input vector"""
        if len(input_vect) != self.input_size:
            raise ValueError("Input size is not correct")

        circuit = QuantumCircuit(self.input_size)

        for _ in range(self.depth):
            for i in range(self.input_size):
                circuit.add_gate(RotX(i, input_vect[i]))

            for i in range(self.input_size - 1):
                circuit.add_gate(CZ(i, i + 1))
            if self.input_size > 1:
                circuit.add_gate(CZ(self.input_size - 1, 0))
        return circuit

    def get_encoding_state(self, input_vect: NDArray[np.double]) -> QuantumState:
        """Returns the encoding state for the given input vector"""
        state = QuantumState(self.input_size)
        state.set_zero_state()
        circuit = self.get_circuit(input_vect)
        circuit.update_quantum_state(state)
        return state

    def __len__(self) -> int:
        """Returns the input size of the encoder"""
        return self.input_size

    def print_circuit(self) -> None:
        """Prints the circuit diagram of the reservoir"""
        circuit = self.get_circuit(np.random.uniform(size=self.input_size))
        circuit_drawer(circuit)
