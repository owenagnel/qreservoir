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

    def __init__(
        self, feature_num: int, depth: int = 1, qubits_per_feature: int = 1
    ) -> None:
        self.feature_num = feature_num
        self.qubit_num = feature_num * qubits_per_feature
        self.qubits_per_feature = qubits_per_feature
        self.depth = depth

    def get_circuit(self, input_vect: NDArray[np.double]) -> QuantumCircuit:
        """Generates the parameterised circuit for the given input vector"""
        if len(input_vect) != self.feature_num:
            raise ValueError("Input size is not correct")

        circuit = QuantumCircuit(self.qubit_num)

        for _ in range(self.depth):
            for i in range(self.feature_num):
                # In the case there are multiple encoding qubits per feature
                for j in range(self.qubits_per_feature):
                    qubit_acted_on = i * self.qubits_per_feature + j
                    circuit.add_gate(RotX(qubit_acted_on, input_vect[i]))

            for i in range(self.qubit_num - 1):
                circuit.add_gate(CZ(i, i + 1))
            if self.qubit_num > 1:
                circuit.add_gate(CZ(self.qubit_num - 1, 0))
        return circuit

    def get_encoding_state(self, input_vect: NDArray[np.double]) -> QuantumState:
        """Returns the encoding state for the given input vector of shape (num_features,)"""
        state = QuantumState(self.qubit_num)
        state.set_zero_state()
        circuit = self.get_circuit(input_vect)
        circuit.update_quantum_state(state)
        return state

    def __len__(self) -> int:
        """Returns the input size of the encoder"""
        return self.qubit_num

    def print_circuit(self) -> None:
        """Prints the circuit diagram of the reservoir"""
        circuit = self.get_circuit(np.random.uniform(size=self.feature_num))
        circuit_drawer(circuit)

    def get_feature_num(self) -> int:
        return self.feature_num
