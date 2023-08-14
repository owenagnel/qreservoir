from qulacs import QuantumCircuit, QuantumState, QuantumGateBase
from qulacs.gate import CZ, RotX, RotY, RotZ
from numpy.typing import NDArray
import numpy as np
from qulacsvis import circuit_drawer
from qreservoir.encoders.Encoder import Encoder


class CHEEncoder(Encoder):
    """Simple Hardware Effecient Encoder class. Implements a
    reuploading stretegy using CZ and random pauli rotation gates
    with a non-cyclic entangling structure"""

    def __init__(self, input_size: int, depth: int) -> None:
        self.input_size = input_size
        self.depth = depth
        # generates fixed random rotation axis for each qubit and depth
        self.gates = ["X", "Y", "Z"]
        self.rotation_gates = np.random.choice(self.gates, (depth, input_size))

    def get_circuit(self, input_vect: NDArray[np.double]) -> QuantumCircuit:
        """Generates the parameterised circuit for the given input vector"""

        if len(input_vect) != self.input_size:
            raise ValueError("Input size is not correct")

        circuit = QuantumCircuit(self.input_size)

        for d in range(self.depth):
            for i in range(self.input_size):
                gate = self.get_random_gate(i, input_vect[i], d)
                circuit.add_gate(gate)

            for i in range(self.input_size - 1):
                circuit.add_gate(CZ(i, i + 1))
            # Uncomment the following line to make the circuit cyclic
            # circuit.add_gate(CZ(self.input_size-1, 0))
        return circuit

    def get_random_gate(self, qubit: int, angle: int, depth: int) -> QuantumGateBase:
        """Returns a random rotation gate on the target qubit"""
        gate = self.rotation_gates[depth, qubit]
        if gate == "X":
            return RotX(qubit, angle)
        elif gate == "Y":
            return RotY(qubit, angle)
        elif gate == "Z":
            return RotZ(qubit, angle)
        else:
            raise ValueError("Gate not recognised")

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
