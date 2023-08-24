from qulacs import QuantumCircuit, QuantumState, QuantumGateBase
from qulacs.gate import CZ, RotX, RotY, RotZ
from numpy.typing import NDArray
import numpy as np
from qulacsvis import circuit_drawer
from qreservoir.encoders.Encoder import Encoder


class ExpEncoder(Encoder):
    """Exponential Encoder class. Implements an expoential encoding scheme stretegy u
    sing CZ and pauli-X rotation gates with a cyclic CZ entangling structure. Typically we
    have multiple qubits per feature, and single depth."""

    def __init__(
        self, feature_num: int, depth: int = 1, qubits_per_feature: int = 1
    ) -> None:
        self.qubits_per_feature = qubits_per_feature
        self.feature_num = feature_num
        self.qubit_num = qubits_per_feature * feature_num
        self.depth = depth

    def get_circuit(self, input_vect: NDArray[np.double]) -> QuantumCircuit:
        """Generates the parameterised circuit for the given input vector"""

        if len(input_vect) != self.feature_num:
            raise ValueError(
                f"Input size is not correct. Expecting vector of size ({self.feature_num},) but got vector of size {input_vect.shape}"
            )

        circuit = QuantumCircuit(self.qubit_num)
        for _ in range(self.depth):
            for f in range(self.feature_num):
                for i in range(self.qubits_per_feature):
                    rot_angle = (3**i) * input_vect[f]
                    circuit.add_gate(RotX(i, rot_angle))
            for i in range(self.qubit_num - 1):
                circuit.add_gate(CZ(i, i + 1))
            if self.qubit_num > 1:
                circuit.add_gate(CZ(self.qubit_num - 1, 0))
        return circuit

    def get_encoding_state(self, input_vect: NDArray[np.double]) -> QuantumState:
        """Returns the encoding state for the given input vector of shape (feature_num,)"""
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
        """Returns the input size of the encoder"""
        return self.feature_num
