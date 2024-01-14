from typing import List

import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumCircuit, QuantumGateBase, QuantumState
from qulacs.gate import CZ, RotX, RotY, RotZ
from qulacsvis import circuit_drawer

from qreservoir.abstract_base_classes import Encoder


class TensorProdEncoder(Encoder):
    """Simple Tensor Product Encoder class.

    Implements a reuploading strategy using only random single-qubit pauli rotation gates
    with no entangling gates. Optionally multiple qubits per feature."""

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
        r"""Initialises the TensorProdEncoder with parameters.

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
