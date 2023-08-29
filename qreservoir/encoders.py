from typing import List

import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumCircuit, QuantumGateBase, QuantumState
from qulacs.gate import CPTP, CZ, RotX, RotY, RotZ, SparseMatrix
from qulacsvis import circuit_drawer
from scipy.sparse import csc_matrix

from qreservoir.abstract_base_classes import Encoder


class CHEEncoder(Encoder):
    """Simple Hardware Effecient Encoder class.

    Implements a reuploading stretegy using CZ and random pauli rotation gates
    with a cyclic entangling structure. Optionally multpile qubits per feature."""

    feature_num: int
    """@private
    number of features to be encoded"""

    qubits_per_feature: int
    """@private
    number of qubits per feature"""

    depth: int
    """@private
    depth of encoding circuit"""

    qubit_num: int
    """@private
    total number of qubits (i.e., `feature_num` * `qubits_per_feature`)"""

    gates: List[str]
    """@private
    Available rotation gates."""

    rotation_gates: NDArray[np.str_]
    """@private
    Fixed array of rotation gates for each qubit and depth"""

    def __init__(
        self, feature_num: int, depth: int = 1, qubits_per_feature: int = 1
    ) -> None:
        r"""Initialises the CHEEncoder with parameters.

        Parameters
        ----------
        feature_num : int
            number of features for each input

        depth : int, optional
            Depth of encoding circuit (number of rotation and entaglement layer pairs), default is 1

        qubits_per_feature: int, optional
            number of qubits onto which each feature is encoded, default is 1

        """
        self.feature_num = feature_num
        self.qubit_num = feature_num * qubits_per_feature
        self.qubits_per_feature = qubits_per_feature
        self.depth = depth
        # generates fixed random rotation axis for each qubit and depth
        self.gates = ["X", "Y", "Z"]
        self.rotation_gates = np.random.choice(
            self.gates, (depth, self.qubit_num)
        )  # makes the encoder return the same circuit each time it is called

    def get_circuit(self, input_vect: NDArray[np.double]) -> QuantumCircuit:
        """Generates the parameterised circuit for the given input vector

        Parameters
        ---------
        input_vect: NDArray[np.double]
            Should be a 1d array with shape (`feature_num`,)

        Returns
        -------
        QuantumCircuit
            Encoding circuit
        """

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

    def get_random_gate(self, qubit: int, angle: float, depth: int) -> QuantumGateBase:
        """Returns a pseudo-random rotation gate on the target qubit (pseudo-random because the rotation gates are fixed for each qubit and depth at object instantiation).

        Parameters
        ----------
        qubit: int
            Qubit to act on

        angle: float
            Angle of rotation

        depth: int
            Depth of encoding circuit

        Returns
        -------
        QuantumGateBase
            Pseudo-random rotation gate
        """
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
        """Returns the encoding state for the given input vector of shape (`num_features`,)

        Parameters
        ----------
        input_vect: NDArray[np.double]
            Input vector, should be a 1d array with shape (`feature_num`,)

        Returns
        -------
        QuantumState
            Encoding state
        """

        state = QuantumState(self.qubit_num)
        state.set_zero_state()
        circuit = self.get_circuit(input_vect)
        circuit.update_quantum_state(state)
        return state

    def __len__(self) -> int:
        """Returns the input size of the encoder

        Returns
        -------
        int
            Number of qubits in encoding circuit"""
        return self.qubit_num

    def print_circuit(self) -> None:
        """Prints the circuit diagram of the reservoir using `qulacsvis` package."""
        circuit = self.get_circuit(np.random.uniform(size=self.feature_num))
        circuit_drawer(circuit)

    def get_feature_num(self) -> int:
        """Returns the input size of the encoder"""
        return self.feature_num


class ExpEncoder(Encoder):
    """Exponential Encoder class.

    Implements an expoential encoding scheme stretegy using CZ and pauli-X rotation gates with a cyclic CZ entangling structure. Typically we
    have multiple qubits per feature, and single depth."""

    feature_num: int
    """@private
    number of features to be encoded"""

    qubits_per_feature: int
    """@private
    number of qubits per feature"""

    depth: int
    """@private
    depth of encoding circuit"""

    def __init__(
        self, feature_num: int, depth: int = 1, qubits_per_feature: int = 1
    ) -> None:
        r"""Initialises the ExpEncoder with parameters.

        Parameters
        ----------
        feature_num : int
            number of features for each input

        depth : int, optional
            Depth of encoding circuit (number of rotation and entaglement layer pairss), default is 1

        qubits_per_feature: int, optional
            number of qubits onto which each feature is encoded, default is 1
        """
        self.qubits_per_feature = qubits_per_feature
        self.feature_num = feature_num
        self.qubit_num = qubits_per_feature * feature_num
        self.depth = depth

    def get_circuit(self, input_vect: NDArray[np.double]) -> QuantumCircuit:
        """Generates the parameterised circuit for the given input vector of shape (`feature_num`,)

        Parameters
        ---------
        input_vect: NDArray[np.double]
            Input vector, should be a 1d array with shape (`feature_num`,).

        Returns
        -------
        QuantumCircuit
            Encoding circuit.
        """

        if len(input_vect) != self.feature_num:
            raise ValueError(
                f"Input size is not correct. Expecting vector of size ({self.feature_num},) but got vector of size {input_vect.shape}"
            )

        circuit = QuantumCircuit(self.qubit_num)
        for _ in range(self.depth):
            for i in range(self.feature_num):
                for j in range(self.qubits_per_feature):
                    rot_angle = (3**j) * input_vect[i]
                    qubit_acted_on = i * self.qubits_per_feature + j
                    circuit.add_gate(RotX(qubit_acted_on, rot_angle))
            for i in range(self.qubit_num - 1):
                circuit.add_gate(CZ(i, i + 1))
            if self.qubit_num > 1:
                circuit.add_gate(CZ(self.qubit_num - 1, 0))
        return circuit

    def get_encoding_state(self, input_vect: NDArray[np.double]) -> QuantumState:
        """Returns the encoding state for the given input vector of shape (`feature_num`,)

        Parameters
        ----------
        input_vect: NDArray[np.double]
            Input vector, should be a 1d array with shape (`feature_num`,)

        Returns
        -------
        QuantumState
            Encoding state
        """
        state = QuantumState(self.qubit_num)
        state.set_zero_state()
        circuit = self.get_circuit(input_vect)
        circuit.update_quantum_state(state)
        return state

    def __len__(self) -> int:
        """Returns the input size of the encoder

        Returns
        -------
        int
            Number of qubits in encoding circuit"""
        return self.qubit_num

    def print_circuit(self) -> None:
        """Prints the circuit diagram of the reservoir using `qulacsvis` package."""
        circuit = self.get_circuit(np.random.uniform(size=self.feature_num))
        circuit_drawer(circuit)

    def get_feature_num(self) -> int:
        """Returns the input size of the encoder

        Returns
        -------
        int
            Number of features to be encoded"""
        return self.feature_num


class HEEncoder(Encoder):
    """Simple Hardware Effecient Encoder class.

    Implements a reuploading stretegy using CZ and RX gates with a cyclic
    entangling structure"""

    feature_num: int
    """@private
    number of features to be encoded"""

    qubits_per_feature: int
    """@private
    number of qubits per feature"""

    depth: int
    """@private
    depth of encoding circuit"""

    qubit_num: int
    """@private
    total number of qubits (i.e., `feature_num` * `qubits_per_feature`)"""

    def __init__(
        self, feature_num: int, depth: int = 1, qubits_per_feature: int = 1
    ) -> None:
        r"""Initialises the HEEncoder with parameters.

        Parameters
        ----------
        feature_num : int
            number of features for each input

        depth : int, optional
            Depth of encoding circuit (number of rotation and entaglement layer pairs), default is 1

        qubits_per_feature: int, optional
            number of qubits onto which each feature is encoded, default is 1
        """
        self.feature_num = feature_num
        self.qubit_num = feature_num * qubits_per_feature
        self.qubits_per_feature = qubits_per_feature
        self.depth = depth

    def get_circuit(self, input_vect: NDArray[np.double]) -> QuantumCircuit:
        """Generates the parameterised circuit for the given input vector of shape (`feature_num`,)

        Parameters
        ---------
        input_vect: NDArray[np.double]
            Input vector, should be a 1d array with shape (`feature_num`,)

        Returns
        -------
        QuantumCircuit
            Encoding circuit
        """

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
        """Returns the encoding state for the given input vector of shape (`feature_num`,)

        Parameters
        ----------
        input_vect: NDArray[np.double]
            Input vector, should be a 1d array with shape (`feature_num`,)

        Returns
        -------
        QuantumState
            Encoding state
        """

        state = QuantumState(self.qubit_num)
        state.set_zero_state()
        circuit = self.get_circuit(input_vect)
        circuit.update_quantum_state(state)
        return state

    def __len__(self) -> int:
        """Returns the input size of the encoder

        Returns
        -------
        int
            Number of qubits in encoding circuit"""
        return self.qubit_num

    def print_circuit(self) -> None:
        """Prints the circuit diagram of the reservoir using `qulacsvis` package."""
        circuit = self.get_circuit(np.random.uniform(size=self.feature_num))
        circuit_drawer(circuit)

    def get_feature_num(self) -> int:
        """Returns the input size of the encoder

        Returns
        -------
        int
            Number of features to be encoded"""
        return self.feature_num


class Noisy_CHEEncoder(Encoder):
    """Noisy CHEEncoder class.

    Implements an complex hardware efficient encoding with Pauli noise between each encoding circuit
    """

    feature_num: int
    """@private
    number of features to be encoded"""

    num_unitaries: int
    """@private
    number of unitaries (i.e. encoding circuits) to be applied"""

    qubits_per_feature: int
    """@private
    number of qubits per feature"""

    depth: int
    """@private
    depth of each unitary"""

    qubit_num: int
    """@private
    total number of qubits (`feature_num` * `qubits_per_feature`)"""

    noise: NDArray[np.double]
    """@private
    noise parameters"""

    unitaries: List[Encoder]
    """@private
    encoding unitary circuits"""

    def __init__(
        self,
        feature_num: int,
        depth: int,
        num_unitaries: int = 1,
        qubits_per_feature: int = 1,
        noise: NDArray[np.double] = np.array([1, 1, 1]) / 3**0.5,
    ) -> None:
        r"""Initialises the Noisy CHEEncoder with parameters.

        Parameters
        ----------
        feature_num : int
            number of features for each input

        depth : int
            depth of encoding circuits

        num_unitaries: int, optional
            number of CHEEncoder unitaries, default is 1

        qubits_per_feature: int, optional
            number of qubits onto which each feature is encoded, default is 1

        noise: NDArray[np.double], optional
            Noise parameters. Should be an np.array of size 3 (one value for each Pauli)
        """
        self.feature_num = feature_num
        self.num_unitaries = num_unitaries
        self.qubits_per_feature = qubits_per_feature
        self.depth = depth
        self.qubit_num = feature_num * qubits_per_feature
        self.noise = noise  # coefficients of single-qubit Kraus operators

        self.unitaries = [
            CHEEncoder(self.feature_num, self.depth, self.qubits_per_feature)
            for _ in range(self.num_unitaries)
        ]

        if len(self.noise) != 3:  # TODO: add check for valid noise parameters
            raise ValueError("Pauli noise needs 3 noise parameters")

    def get_circuit(self, input_vect: NDArray[np.double]) -> QuantumCircuit:
        """Generates the noisy circuit for the given input array

        Parameters
        ---------
        input: NDArray[np.double]
            Should be a 2d array with shape (`num_unitaries`, `feature_num`)

        Returns
        -------
        QuantumCircuit
            Noisy encoding circuit
        """
        if input_vect.shape[0] != self.num_unitaries:
            raise ValueError(
                "First dimension of input should correspond to number of unitary layers"
            )

        circuit = QuantumCircuit(self.qubit_num)
        self.apply_noise_channel(circuit)

        for i, unitary in enumerate(self.unitaries):
            unitary_circuit = unitary.get_circuit(input_vect[i])
            circuit.merge_circuit(unitary_circuit)
            self.apply_noise_channel(circuit)

        return circuit

    def apply_noise_channel(self, circuit: QuantumCircuit) -> None:
        """Function to apply one layer of noise to the passed circuit.

        Parameters
        ----------
        circuit: QuantumCircuit
            circuit to append noise layer to
        """

        qx, qy, qz = self.noise[0], self.noise[1], self.noise[2]
        X, Y, Z = (
            csc_matrix([[0, qx], [qx, 0]]),
            csc_matrix([[0.0, -1j * qy], [1j * qy, 0]]),
            csc_matrix([[qz, 0], [0, -qz]]),
        )

        for i in range(self.qubit_num):
            kraus_ops = [
                SparseMatrix([i], X),
                SparseMatrix([i], Y),
                SparseMatrix([i], Z),
            ]
            single_noise_op = CPTP(kraus_ops)
            circuit.add_gate(single_noise_op)

    def get_encoding_state(self, input_vect: NDArray[np.double]) -> QuantumState:
        """Returns the encoding state for the given input array

        Parameters
        ----------
        input_vect: NDArray[np.double]
            Should be a 2d array with shape (`num_unitaries`, `feature_num`).

        Returns
        -------
        QuantumState
            Encoding state
        """

        state = QuantumState(self.qubit_num)
        state.set_zero_state()
        circuit = self.get_circuit(input_vect)
        circuit.update_quantum_state(state)

        return state

    def __len__(self) -> int:
        """Returns the number of qubits of the encoder

        Returns
        -------
        int
            Number of qubit sin encoding circuit"""
        return self.qubit_num

    def print_circuit(self) -> None:
        """Prints the circuit diagram of the reservoir using `qulacsvis` package"""
        circuit = self.get_circuit(
            np.random.uniform(size=(self.num_unitaries, self.feature_num))
        )
        circuit_drawer(circuit)

    def get_feature_num(self) -> int:
        """Returns the input size to the encoder

        Returns
        -------
        int
            Number of features to be encoded"""
        return self.feature_num


class NonCorrelatedCHEE(Encoder):
    r"""Non correlated case of CHE Encoder class.

    Each layer encodes a different set of features. Hence get_encoding_state() takes an input of shape (`depth`, `feature_num`).

    .. deprecated:: 0.3.0
        `NonCorrelatedCHEE` was used for testing only and doesn't fit the with the standard `Encoder` interface. Use `CHEEncoder` instead.
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
