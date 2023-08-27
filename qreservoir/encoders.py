import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumCircuit, QuantumGateBase, QuantumState
from qulacs.gate import CZ, RotX, RotY, RotZ
from qulacsvis import circuit_drawer

from qreservoir.abstract_base_classes import Encoder


class CHEEncoder(Encoder):
    """Simple Hardware Effecient Encoder class. Implements a
    reuploading stretegy using CZ and random pauli rotation gates
    with a cyclic entangling structure. Optionally multpile qubits per feature."""

    def __init__(
        self, feature_num: int, depth: int = 1, qubits_per_feature: int = 1
    ) -> None:
        self.feature_num = feature_num
        self.qubit_num = feature_num * qubits_per_feature
        self.qubits_per_feature = qubits_per_feature
        self.depth = depth
        # generates fixed random rotation axis for each qubit and depth
        self.gates = ["X", "Y", "Z"]
        self.rotation_gates = np.random.choice(self.gates, (depth, self.qubit_num))

    def get_circuit(self, input_vect: NDArray[np.double]) -> QuantumCircuit:
        """Generates the parameterised circuit for the given input vector"""

        if len(input_vect) != self.feature_num:
            raise ValueError("Input size is not correct")

        circuit = QuantumCircuit(self.qubit_num)

        for d in range(self.depth):
            for i in range(self.feature_num):
                # In the case there are multiple encoding qubits per feature
                for j in range(self.qubits_per_feature):
                    qubit_acted_on = i * self.qubits_per_feature + j
                    gate = self.get_random_gate(qubit_acted_on, input_vect[i], d)
                    circuit.add_gate(gate)

            for i in range(self.qubit_num - 1):
                circuit.add_gate(CZ(i, i + 1))
            if self.qubit_num > 1:
                circuit.add_gate(CZ(self.qubit_num - 1, 0))
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
        """Returns the input size of the encoder"""
        return self.feature_num


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


class NonCorrelatedCHEE(Encoder):
    """Non correlated case of CHE Encoder class. Each layer encodes a different set of features. Hence get_encoding_state() takes an input of shape (`depth`, `feature_num`).
    .. deprecated
    """

    def __init__(
        self, feature_num: int, depth: int = 1, qubits_per_feature: int = 1
    ) -> None:
        self.feature_num = feature_num
        self.qubit_num = feature_num * qubits_per_feature
        self.qubits_per_feature = qubits_per_feature
        self.depth = depth
        # generates fixed random rotation axis for each qubit and depth
        self.gates = ["X", "Y", "Z"]
        self.rotation_gates = np.random.choice(self.gates, (depth, self.qubit_num))

    def get_circuit(self, input_vect: NDArray[np.double]) -> QuantumCircuit:
        """Generates the parameterised circuit for the given input vector"""

        circuit = QuantumCircuit(self.qubit_num)

        for d in range(self.depth):
            for i in range(self.feature_num):
                # In the case there are multiple encoding qubits per feature
                for j in range(self.qubits_per_feature):
                    qubit_acted_on = i * self.qubits_per_feature + j
                    gate = self.get_random_gate(qubit_acted_on, input_vect[d][i], d)
                    circuit.add_gate(gate)

            for i in range(self.qubit_num - 1):
                circuit.add_gate(CZ(i, i + 1))
            if self.qubit_num > 1:
                circuit.add_gate(CZ(self.qubit_num - 1, 0))
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
        """Returns the encoding state for the given input vector of shape (num_features,)"""
        if input_vect.shape != (self.depth, self.feature_num):
            raise ValueError(
                f"Expected input_vect of shape {(self.depth, self.feature_num)}, got {input_vect.shape}"
            )
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
