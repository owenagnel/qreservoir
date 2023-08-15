from qulacs import QuantumCircuit, QuantumState, NoiseSimulator, SimulationResult
import numpy as np
from numpy.typing import NDArray
from qulacsvis import circuit_drawer
from qulacs.gate import RotX, CZ
from qreservoir.encoders.Encoder import Encoder


class NoisyHEEncoder(Encoder):
    """Simple Noisy Hardware Effecient Encoder class. Implements
    a reuploading stretegy using CZ and RX gates with a cyclic
    entangling structure with depolarizing noise by default.

    WORK IN PROGRESS"""

    # def __init__(
    #     self,
    #     input_size: int,
    #     depth: int,
    #     noise_type: str = "Depolarizing",
    #     noise_probability: float = 0.01,
    #     sample_size: int = 100,
    # ) -> None:
    #     """Generates the parameterised circuit for the given input vector"""
    #     self.feature_num = input_size
    #     self.qubit_num = input_size
    #     self.depth = depth
    #     self.noise_type = noise_type
    #     self.noise_probability = noise_probability
    #     self.sample_size = sample_size

    # def get_circuit(self, input_vect: NDArray[np.double]) -> QuantumCircuit:
    #     """Returns the encoding state for the given input vector"""

    #     if len(input_vect) != self.feature_num:
    #         raise ValueError("Input size is not correct")

    #     circuit = QuantumCircuit(self.qubit_num)

    #     for _ in range(self.depth):
    #         for i in range(self.qubit_num):
    #             circuit.add_noise_gate(
    #                 RotX(i, input_vect[i]), self.noise_type, self.noise_probability
    #             )
    #         for i in range(self.qubit_num - 1):
    #             circuit.add_noise_gate(
    #                 CZ(i, i + 1), self.noise_type, self.noise_probability
    #             )
    #         if self.qubit_num > 1:
    #             circuit.add_noise_gate(
    #                 CZ(self.qubit_num - 1, 0), self.noise_type, self.noise_probability
    #             )
    #     return circuit

    # def get_encoding_state(self, input_vect: NDArray[np.double]) -> QuantumState:
    #     """Returns the encoding state for the given input vector of shape (num_features,)"""
    #     init_state = QuantumState(self.qubit_num)
    #     init_state.set_zero_state()
    #     circuit = self.get_circuit(input_vect)
    #     sim = NoiseSimulator(circuit, init_state)
    #     result = sim.execute_and_get_result(self.sample_size)
    #     return result.get_state()

    # def __len__(self) -> int:
    #     """Returns the input size of the encoder"""
    #     return self.qubit_num

    # def print_circuit(self) -> None:
    #     """Prints the circuit diagram of the reservoir"""
    #     circuit = self.get_circuit(np.random.uniform(size=self.feature_num))
    #     circuit_drawer(circuit)

    # def get_feature_num(self) -> int:
    #     """Returns the input size of the encoder"""
    #     return self.feature_num
