from qulacs import QuantumCircuit, QuantumState, QuantumGateBase, DensityMatrix
from qulacs.state import tensor_product
from qulacs.gate import DenseMatrix, X, Z
import numpy as np
from numpy.typing import NDArray
from typing import Union, cast
from qulacsvis import circuit_drawer
from qreservoir.encoders.Encoder import Encoder
from qreservoir.reservoirs.Reservoir import Reservoir
from functools import reduce
from typing import List, Tuple


class IsingMagTraverseReservoir(Reservoir):
    """The CNOTReservoir class simulates a reservoir of dynamics.
    The reservoir is made up of CNOT gates in a cyclic entangling structure"""

    def __init__(
        self,
        encoder: Union[Encoder, None],
        ancilla_num: int,
        enc_qubit_num: int = 5,
        time_step: float = 0.5,
    ) -> None:
        """Initialises the reservoir with the correct number of qubits
        given an encoder and ancilla qubit number. If no encoder is provided,
        enc_qubit_num is used to initialise the reservoir.
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
        ham = np.zeros((2**self.total_size, 2**self.total_size), dtype=complex)
        for i in range(self.total_size):  ## i runs 0 to nqubit-1
            Jx = -1.0 + 2.0 * np.random.rand()  ## random number in -1~1
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
        """Take list_SiteAndOperator = [ [i_0, O_0], [i_1, O_1], ...],
        Insert Identity into unrelated qubit
        make (2**nqubit, 2**nqubit) matrix:
        I(0) * ... * O_0(i_0) * ... * O_1(i_1) ..."""

        list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
        list_SingleGates = []  ## Arrange 1-qubit gates and reduce with np.kron
        cnt = 0
        for i in range(self.total_size):
            if i in list_Site:
                list_SingleGates.append(list_SiteAndOperator[cnt][1])
                cnt += 1
            else:  ## an empty site is identity
                list_SingleGates.append(self.I_mat)

        return reduce(np.kron, list_SingleGates)

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

            if type(prev) == QuantumState:
                full_state = tensor_product(prev, encoding_state)

            elif type(prev) == DensityMatrix:
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
