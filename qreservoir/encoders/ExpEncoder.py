from qulacs import QuantumCircuit, QuantumState, QuantumGateBase
from qulacs.gate import CZ, RotX, RotY, RotZ
from numpy.typing import NDArray
import numpy as np
from qulacsvis import circuit_drawer
from qreservoir.encoders.Encoder import Encoder


class ExpEncoder(Encoder):
    """TODO: Write me"""

    def __init__(self, qubit_num: int, feature_num: int = 0) -> None:
        self.qubit_num = qubit_num
        self.feature_num = feature_num

    def get_circuit(self, input_vect: NDArray[np.double]) -> QuantumCircuit:
        """Generates the parameterised circuit for the given input vector"""

        if len(input_vect) != self.feature_num:
            raise ValueError("Input size is not correct")

        circuit = QuantumCircuit(self.qubit_num)

        for f in range(self.feature_num):
            for i in range(self.qubit_num):
                rot_angle = 3 ** (i - 1) * input_vect[f]
                circuit.add_gate(RotX(i, rot_angle))
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
