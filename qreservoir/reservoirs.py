from functools import reduce
from typing import List, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from qulacs import DensityMatrix, QuantumCircuit, QuantumGateBase, QuantumState
from qulacs.gate import CNOT, CZ, DenseMatrix, RandomUnitary, RotX, RotY, RotZ, X, Z
from qulacs.state import tensor_product
from qulacsvis import circuit_drawer

from qreservoir.abstract_base_classes import Encoder, Reservoir


class CNOTReservoir(Reservoir):
    r"""The CNOT class simulates a reservoir with CNOT dynamics.
    The reservoir is made up of CNOT gates in a cyclic
    entangling structure.

    The reservoir structure is:
    .. math:: \prod_{l=1}^d \text{C-Phase}_{N,1} \prod_{i=1}^{n-1} \text{C-Phase}_{i,i+1}
    """

    encoder: Optional[Encoder]
    """
    @private
    The encoder used to encode the input data into a quantum state"""

    enc_qubit_num: int
    """
    @private
    The number of qubits used to encode the input data"""

    ancilla_num: int
    """
    @private
    The number of ancilla qubits"""

    total_size: int
    """
    @private
    The total number of qubits in the reservoir"""

    depth: int
    """
    @private
    The depth of the reservoir"""

    dynamics_circuit: QuantumCircuit
    """
    @private
    The circuit representing the reservoir dynamics"""

    def __init__(
        self,
        encoder: Union[Encoder, None],
        ancilla_num: int,
        depth: int,
        enc_qubit_num: int = 5,
    ) -> None:
        r"""Initialises the reservoir with the correct number of qubits
        given an encoder and ancilla qubit number. If no encoder is provided,
        `enc_qubit_num` is used to initialise the reservoir.

        Parameters
        ----------
        encoder : Encoder, optional
            An encoder object to be used in tandem with the reservoir.
        ancilla_num : int
            The number of qubits in the hidden space of our reservoir.
            These qubits qre unaffected by the input data before the reservoir
            dynamics take place. In a reservoir model they are what allows for the memory property.
        depth : int
            Depth of the reservoir.

        Other Parameters
        ----------------
        enc_qubit_num : int, optional
            Infrequently used paramter to manually specify encoding qubit number.
            Allows us to initialise the reservoir without an encoder. If an encoder is provided, this
            argument is ignored.
        """

        self.encoder = encoder
        self.enc_qubit_num = (
            enc_qubit_num if encoder is None else len(encoder)
        )  # bit skecthy... maybe you should check if encoder is None and then set enc_qubit_num to 0
        self.ancilla_num = ancilla_num
        self.total_size = self.enc_qubit_num + self.ancilla_num
        self.depth = depth
        self.dynamics_circuit = self.get_dynamics_circuit()

    def get_dynamics_circuit(self) -> QuantumCircuit:
        """Constructs a random dynamics circuit"""
        circuit = QuantumCircuit(self.total_size)
        for _ in range(self.depth):
            for i in range(self.total_size - 1):
                circuit.add_gate(CNOT(i, i + 1))
            if self.total_size > 1:
                circuit.add_gate(CNOT(self.total_size - 1, 0))
        return circuit

    def get_reservoir_state(
        self,
        input_vect: NDArray[np.double],
        prev: Union[QuantumState, DensityMatrix, None] = None,
    ) -> Union[QuantumState, DensityMatrix]:
        """Returns the final state of the reservoir after encoding the input
        and applying the dynamics; if prev is passed, ouput will match the
        type of prev (i.e. QuantumState or DensityMatrix)"""

        if self.encoder is None:
            raise ValueError("Encoder must be connected before getting reservoir state")

        cast(Encoder, self.encoder)  # cast to Encoder type so mypy is happy
        encoding_state = self.encoder.get_encoding_state(input_vect)

        if prev:
            if prev.get_qubit_count() != self.ancilla_num:
                raise ValueError("prev must have the correct number of ancilla qubits")

            if isinstance(prev, QuantumState):
                full_state = tensor_product(prev, encoding_state)

            elif isinstance(prev, DensityMatrix):
                # Cast encoding state as a density matrix
                encoding_state_as_density_matrix = DensityMatrix(self.enc_qubit_num)
                encoding_state_as_density_matrix.load(encoding_state)
                # Calculate tensor of both hidden and accessible systems
                full_state = tensor_product(prev, encoding_state_as_density_matrix)

            else:
                raise ValueError("prev must be a QuantumState or DensityMatrix")

        else:
            ancilla_state = QuantumState(self.ancilla_num)
            ancilla_state.set_zero_state()
            full_state = tensor_product(ancilla_state, encoding_state)

        self.dynamics_circuit.update_quantum_state(full_state)
        return full_state

    def connect_encoder(self, encoder: Encoder) -> None:
        """Connects an encoder to the reservoir."""
        if len(encoder) != self.enc_qubit_num:
            raise ValueError("Cannot switch with encoder of a different input size")
        self.encoder = encoder

    def __len__(self) -> int:
        """Returns the number of qubits in the reservoir"""
        return self.total_size

    def print_circuit(self) -> None:
        """Prints the circuit diagram of the reservoir"""
        circuit_drawer(self.dynamics_circuit)

    def get_ancilla_num(self) -> int:
        """Returns the number of ancilla qubits"""
        return self.ancilla_num

    def get_encoding_qubit_num(self) -> int:
        """Returns the number of input qubits"""
        return self.enc_qubit_num


class HarrRandomReservoir(Reservoir):
    r"""The HarrRandomReservoir class simulates a reservoir with a random unitary as dynamics.
    The reservoir is made up of a single unitary sampled from the haar measure.
    """

    encoder: Optional[Encoder]
    """
    @private
    The encoder used to encode the input data into a quantum state"""

    enc_qubit_num: int
    """
    @private
    The number of qubits used to encode the input data"""

    ancilla_num: int
    """
    @private
    The number of ancilla qubits"""

    total_size: int
    """
    @private
    The total number of qubits in the reservoir"""

    dynamics_circuit: QuantumCircuit
    """
    @private
    The circuit representing the reservoir dynamics"""

    def __init__(
        self,
        encoder: Union[Encoder, None],
        ancilla_num: int,
        enc_qubit_num: int = 5,
    ) -> None:
        r"""Initialises the reservoir with the correct number of qubits
        given an encoder and ancilla qubit number. If no encoder is provided,
        `enc_qubit_num` is used to initialise the reservoir.

        Parameters
        ----------
        encoder : Encoder, optional
            An encoder object to be used in tandem with the reservoir.
        ancilla_num : int
            The number of qubits in the hidden space of our reservoir.

        Other Parameters
        ----------------
        enc_qubit_num : int, optional
            Infrequently used paramter to manually specify encoding qubit number.
            Allows us to initialise the reservoir without an encoder. If an encoder is provided, this
            argument is ignored.
        """

        self.encoder = encoder
        self.enc_qubit_num = (
            enc_qubit_num if encoder is None else len(encoder)
        )  # bit skecthy... maybe you should check if encoder is None and then set enc_qubit_num to 0
        self.ancilla_num = ancilla_num
        self.total_size = self.enc_qubit_num + self.ancilla_num
        self.dynamics_circuit = self.get_dynamics_circuit()

    def get_dynamics_circuit(self) -> QuantumCircuit:
        """Constructs a random dynamics circuit"""
        circuit = QuantumCircuit(self.total_size)
        # Add random unitary matrix gate
        harr_random_matrix = RandomUnitary(list(range(self.total_size)))
        circuit.add_gate(harr_random_matrix)
        return circuit

    def get_reservoir_state(
        self,
        input_vect: NDArray[np.double],
        prev: Union[QuantumState, DensityMatrix, None] = None,
    ) -> Union[QuantumState, DensityMatrix]:
        """Returns the final state of the reservoir after encoding `input_vect`
        and applying the dynamics; if prev is passed, ouput will match the
        type of `prev` (i.e. QuantumState or DensityMatrix)"""

        if self.encoder is None:
            raise ValueError("Encoder must be connected before getting reservoir state")

        cast(Encoder, self.encoder)  # cast to Encoder type so mypy is happy
        encoding_state = self.encoder.get_encoding_state(input_vect)

        if prev:
            if prev.get_qubit_count() != self.ancilla_num:
                raise ValueError("prev must have the correct number of ancilla qubits")

            if isinstance(prev, QuantumState):
                full_state = tensor_product(prev, encoding_state)

            elif isinstance(prev, DensityMatrix):
                # Cast encoding state as a density matrix
                encoding_state_as_density_matrix = DensityMatrix(self.enc_qubit_num)
                encoding_state_as_density_matrix.load(encoding_state)
                # Calculate tensor of both hidden and accessible systems
                full_state = tensor_product(prev, encoding_state_as_density_matrix)

            else:
                raise ValueError("prev must be a QuantumState or DensityMatrix")

        else:
            ancilla_state = QuantumState(self.ancilla_num)
            ancilla_state.set_zero_state()
            full_state = tensor_product(ancilla_state, encoding_state)

        self.dynamics_circuit.update_quantum_state(full_state)
        return full_state

    def connect_encoder(self, encoder: Encoder) -> None:
        """Connects an encoder to the reservoir."""
        if len(encoder) != self.enc_qubit_num:
            raise ValueError("Cannot switch with encoder of a different input size")
        self.encoder = encoder

    def __len__(self) -> int:
        """Returns the number of qubits in the reservoir"""
        return self.total_size

    def print_circuit(self) -> None:
        """Prints the circuit diagram of the reservoir"""
        circuit_drawer(self.dynamics_circuit)

    def get_ancilla_num(self) -> int:
        """Returns the number of ancilla qubits"""
        return self.ancilla_num

    def get_encoding_qubit_num(self) -> int:
        """Returns the number of input qubits"""
        return self.enc_qubit_num


class IsingMagTraverseReservoir(Reservoir):
    r"""The IsingMagTraverseReservoir class simulates a reservoir with Ising Hamiltonian dynamics.
    The reservoir is made up of a single ising hamiltonian with random coefficients.

    The reservoir structure is:
    .. math:: U_{\text{Ising}} = \exp(-i H t)

    where

    .. math:: H = \sum_{j=1}^N a_j X_j + \sum_{j=1}^N \sum_{k=1}^{j-1} J_{jk} Z_j Z_k


    Where coefficients :math:`J` and :math:`a` are randomly sampled from the uniform distribution.
    """

    encoder: Optional[Encoder]
    """
    @private
    The encoder used to encode the input data into a quantum state"""

    enc_qubit_num: int
    """
    @private
    The number of qubits used to encode the input data"""

    ancilla_num: int
    """
    @private
    The number of ancilla qubits"""

    total_size: int
    """
    @private
    The total number of qubits in the reservoir"""

    time_step: float
    """
    @private
    The time step of the ising hamiltonian dynamics"""

    I_mat: NDArray[np.double]
    """
    @private
    The identity matrix (helper)"""

    X_mat: NDArray[np.double]
    """
    @private
    The Pauli-X matrix (helper)"""

    Z_mat: NDArray[np.double]
    """
    @private
    The Pauli-Z matrix (helper)"""

    dynamics_circuit: QuantumCircuit
    """
    @private
    The circuit representing the reservoir dynamics"""

    def __init__(
        self,
        encoder: Union[Encoder, None],
        ancilla_num: int,
        enc_qubit_num: int = 5,
        time_step: float = 0.5,
    ) -> None:
        r"""Initialises the reservoir with the correct number of qubits
        given an encoder and ancilla qubit number. If no encoder is provided,
        `enc_qubit_num` is used to initialise the reservoir.

        Parameters
        ----------
        encoder : Encoder, optional
            An encoder object to be used in tandem with the reservoir.
        ancilla_num : int
            The number of qubits in the hidden space of our reservoir.
            These qubits qre unaffected by the input data before the reservoir
            dynamics take place. In a reservoir model they are what allows for the memory property.
        time_step : float, optional
            Time step of the ising hamiltonian dynamics.

        Other Parameters
        ----------------
        enc_qubit_num : int, optional
            Infrequently used paramter to manually specify encoding qubit number.
            Allows us to initialise the reservoir without an encoder. If an encoder is provided, this
            argument is ignored.
        """

        self.encoder = encoder
        self.enc_qubit_num = (
            enc_qubit_num if encoder is None else len(encoder)
        )  # bit skecthy... maybe you should check if encoder is None and then set enc_qubit_num to 0

        self.time_step = time_step
        # helper matrices
        self.I_mat = np.eye(2, dtype=complex)
        self.X_mat = X(0).get_matrix()
        self.Z_mat = Z(0).get_matrix()

        # circuit generation
        self.ancilla_num = ancilla_num
        self.total_size = self.enc_qubit_num + self.ancilla_num
        self.dynamics_circuit = self.get_dynamics_circuit()

    def get_dynamics_circuit(self) -> QuantumCircuit:
        """Constructs a random dynamics circuit"""
        circuit = QuantumCircuit(self.total_size)
        circuit.add_gate(self.get_ising_hamiltonian())
        return circuit

    def get_ising_hamiltonian(self) -> QuantumGateBase:
        """Constructs a random ising hamiltonian gate"""
        ham = np.zeros((2**self.total_size, 2**self.total_size), dtype=complex)
        for i in range(self.total_size):  # i runs 0 to nqubit-1
            Jx = -1.0 + 2.0 * np.random.rand()  # random number in -1~1
            ham += Jx * self.make_fullgate([(i, self.X_mat)])
            for j in range(i + 1, self.total_size):
                J_ij = -1.0 + 2.0 * np.random.rand()
                ham += J_ij * self.make_fullgate([(i, self.Z_mat), (j, self.Z_mat)])

        ## Create a time evolution operator by diagonalization. H*P = P*D <-> H = P*D*P^dagger
        diag, eigen_vecs = np.linalg.eigh(ham)
        time_evol_op = np.dot(
            np.dot(eigen_vecs, np.diag(np.exp(-1j * self.time_step * diag))),
            eigen_vecs.T.conj(),
        )  # e^-iHT
        return DenseMatrix(list(range(self.total_size)), time_evol_op)

    def make_fullgate(
        self, list_SiteAndOperator: List[Tuple[int, np.ndarray]]
    ) -> NDArray[np.double]:
        """Take `list_SiteAndOperator` = [ [i_0, O_0], [i_1, O_1], ...],
        Insert I into unrelated qubit
        make (2**nqubit, 2**nqubit) matrix:
        I(0) * ... * O_0(i_0) * ... * O_1(i_1) ..."""

        list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
        list_SingleGates = []  # Arrange 1-qubit gates and reduce with np.kron
        cnt = 0
        for i in range(self.total_size):
            if i in list_Site:
                list_SingleGates.append(list_SiteAndOperator[cnt][1])
                cnt += 1
            else:  # an empty site is identity
                list_SingleGates.append(self.I_mat)

        return reduce(np.kron, list_SingleGates)

    def get_reservoir_state(
        self,
        input_vect: NDArray[np.double],
        prev: Union[QuantumState, DensityMatrix, None] = None,
    ) -> Union[QuantumState, DensityMatrix]:
        """Returns the final state of the reservoir after encoding `input_vect`
        and applying the dynamics; if `prev` is passed, ouput will match the
        type of `prev` (i.e. `QuantumState` or `DensityMatrix`)"""

        if self.encoder is None:
            raise ValueError("Encoder must be connected before getting reservoir state")

        cast(Encoder, self.encoder)  # cast to Encoder type so mypy is happy
        encoding_state = self.encoder.get_encoding_state(input_vect)

        if prev:
            if prev.get_qubit_count() != self.ancilla_num:
                raise ValueError("prev must have the correct number of ancilla qubits")

            if isinstance(prev, QuantumState):
                full_state = tensor_product(prev, encoding_state)

            elif isinstance(prev, DensityMatrix):
                # Cast encoding state as a density matrix
                encoding_state_as_density_matrix = DensityMatrix(self.enc_qubit_num)
                encoding_state_as_density_matrix.load(encoding_state)
                # Calculate tensor of both hidden and accessible systems
                full_state = tensor_product(prev, encoding_state_as_density_matrix)

            else:
                raise ValueError("prev must be a QuantumState or DensityMatrix")

        else:
            ancilla_state = QuantumState(self.ancilla_num)
            ancilla_state.set_zero_state()
            full_state = tensor_product(ancilla_state, encoding_state)

        self.dynamics_circuit.update_quantum_state(full_state)
        return full_state

    def connect_encoder(self, encoder: Encoder) -> None:
        """Connects an encoder to the reservoir."""
        if len(encoder) != self.enc_qubit_num:
            raise ValueError("Cannot switch with encoder of a different input size")
        self.encoder = encoder

    def __len__(self) -> int:
        """Returns the number of qubits in the reservoir"""
        return self.total_size

    def print_circuit(self) -> None:
        """Prints the circuit diagram of the reservoir"""
        circuit_drawer(self.dynamics_circuit)

    def get_ancilla_num(self) -> int:
        """Returns the number of ancilla qubits"""
        return self.ancilla_num

    def get_encoding_qubit_num(self) -> int:
        """Returns the number of input qubits"""
        return self.enc_qubit_num


class RotationReservoir(Reservoir):
    r"""The RandomReservoir class simulates a reservoir with random dynamics.
    The reservoir is made up of random rotations and CZ gates in a cyclic
    entangling structure.

    The reservoir structure is:
    .. math:: \prod_{l=1}^d W V(\mathbf{k}_l, \mathbf{\theta}_l)

    with

    .. math:: W = \text{C-Phase}_{N,1} \times \prod_{i=1}^{n-1} \text{C-Phase}_{i,i+1}

    and

    .. math:: V(\mathbf{k}_l, \mathbf{\theta}_l) = \prod_{i=1}^n R_{k_i^l}(\theta_i^l)

    Where :math:`R_{k_i^l}(\theta_i^l)` is a rotation of the :math:`i`th qubit by an angle :math:`\theta_i^l`
    about the :math:`k_i^l = x, y` or :math:`z` axis.
    """

    encoder: Optional[Encoder]
    """
    @private
    The encoder used to encode the input data into a quantum state"""

    enc_qubit_num: int
    """
    @private
    The number of qubits used to encode the input data"""

    ancilla_num: int
    """
    @private
    The number of ancilla qubits"""

    total_size: int
    """
    @private
    The total number of qubits in the reservoir"""

    depth: int
    """
    @private
    The depth of the reservoir"""

    gates: List[str]
    """
    @private
    The gate set used in the reservoir dynamics"""

    dynamics_circuit: QuantumCircuit
    """
    @private
    The circuit representing the reservoir dynamics"""

    def __init__(
        self,
        encoder: Optional[Encoder],
        ancilla_num: int,
        depth: int,
        enc_qubit_num: int = 5,
    ) -> None:
        r"""Initialises the reservoir with the correct number of qubits
        given an encoder and ancilla qubit number. If no encoder is provided,
        `enc_qubit_num` is used to initialise the reservoir.

        Parameters
        ----------
        encoder : Encoder, optional
            An encoder object to be used in tandem with the reservoir.
        ancilla_num : int
            The number of qubits in the hidden space of our reservoir.
            These qubits qre unaffected by the input data before the reservoir
            dynamics take place. In a reservoir model they are what allows for the memory property.
        depth : int
            Depth of the reservoir.

        Other Parameters
        ----------------
        enc_qubit_num : int, optional
            Infrequently used paramter to manually specify encoding qubit number.
            Allows us to initialise the reservoir without an encoder. If an encoder is provided, this
            argument is ignored.
        """

        self.encoder = encoder
        self.enc_qubit_num = (
            enc_qubit_num if encoder is None else len(encoder)
        )  # bit skecthy... maybe you should check if encoder is None and then set enc_qubit_num to 0
        self.ancilla_num = ancilla_num
        self.total_size = self.enc_qubit_num + self.ancilla_num
        self.depth = depth
        self.gates = ["X", "Y", "Z"]
        self.dynamics_circuit = self.get_dynamics_circuit()

    def get_dynamics_circuit(self) -> QuantumCircuit:
        """Constructs a random dynamics circuit"""
        circuit = QuantumCircuit(self.total_size)
        for _ in range(self.depth):
            for i in range(self.total_size):
                circuit.add_gate(self.get_random_gate(i))
            for i in range(self.total_size - 1):
                circuit.add_gate(CZ(i, i + 1))
            if self.total_size > 1:
                circuit.add_gate(CZ(self.total_size - 1, 0))
        return circuit

    def get_random_gate(self, target: int) -> QuantumGateBase:
        """Returns a random rotation gate on the target qubit"""
        rotation_gate = np.random.choice(self.gates)
        rangle = np.random.rand() * 2 * np.pi
        if rotation_gate == "X":
            return RotX(target, rangle)
        elif rotation_gate == "Y":
            return RotY(target, rangle)
        elif rotation_gate == "Z":
            return RotZ(target, rangle)
        else:
            raise ValueError("Gate not recognised")

    def get_reservoir_state(
        self,
        input_vect: NDArray[np.double],
        prev: Union[QuantumState, DensityMatrix, None] = None,
    ) -> Union[QuantumState, DensityMatrix]:
        """Returns the final state of the reservoir after encoding the `input_vect`
        and applying the dynamics; if `prev` is passed, ouput will match the
        type of prev (i.e. QuantumState or DensityMatrix)"""

        if self.encoder is None:
            raise ValueError("Encoder must be connected before getting reservoir state")

        cast(Encoder, self.encoder)  # cast to Encoder type so mypy is happy
        encoding_state = self.encoder.get_encoding_state(input_vect)

        if prev:
            if prev.get_qubit_count() != self.ancilla_num:
                raise ValueError("prev must have the correct number of ancilla qubits")

            if isinstance(prev, QuantumState):
                full_state = tensor_product(prev, encoding_state)

            elif isinstance(prev, DensityMatrix):
                # Cast encoding state as a density matrix
                encoding_state_as_density_matrix = DensityMatrix(self.enc_qubit_num)
                encoding_state_as_density_matrix.load(encoding_state)
                # Calculate tensor of both hidden and accessible systems
                full_state = tensor_product(prev, encoding_state_as_density_matrix)

            else:
                raise ValueError("prev must be a QuantumState or DensityMatrix")

        else:
            ancilla_state = QuantumState(self.ancilla_num)
            ancilla_state.set_zero_state()
            full_state = tensor_product(ancilla_state, encoding_state)

        self.dynamics_circuit.update_quantum_state(full_state)
        return full_state

    def connect_encoder(self, encoder: Encoder) -> None:
        """Connects an encoder to the reservoir."""
        if len(encoder) != self.enc_qubit_num:
            raise ValueError("Cannot switch with encoder of a different input size")
        self.encoder = encoder

    def __len__(self) -> int:
        """Returns the number of qubits in the reservoir"""
        return self.total_size

    def print_circuit(self) -> None:
        """Prints the circuit diagram of the reservoir to terminal."""
        circuit_drawer(self.dynamics_circuit)

    def get_ancilla_num(self) -> int:
        """Returns the number of ancilla qubits"""
        return self.ancilla_num

    def get_encoding_qubit_num(self) -> int:
        """Returns the number of input qubits"""
        return self.enc_qubit_num
