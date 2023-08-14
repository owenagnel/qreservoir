from qulacs import QuantumCircuit, QuantumState, QuantumGateBase, DensityMatrix
from qulacs.state import tensor_product
from qulacs.gate import RandomUnitary
import numpy as np
from numpy.typing import NDArray
from typing import Union, cast
from qulacsvis import circuit_drawer
from qreservoir.encoders.Encoder import Encoder
from qreservoir.reservoirs.Reservoir import Reservoir


class HarrRandomReservoir(Reservoir):
    """The HarrRandomReservoir class simulates a reservoir with random dynamics.
    The reservoir is essentially a harr random unitary matrix"""

    def __init__(
        self,
        encoder: Union[Encoder, None],
        ancilla_num: int,
        input_size: int = 5,
    ) -> None:
        """Initialises the reservoir with the correct number of qubits
        given an encoder and ancilla qubit number. If no encoder is provided,
        input_size is used to initialise the reservoir.
        """

        self.encoder = encoder
        self.input_size = (
            input_size if encoder is None else len(encoder)
        )  # bit skecthy... maybe you should check if encoder is None and then set input_size to 0
        self.ancilla_num = ancilla_num
        self.total_size = self.input_size + self.ancilla_num
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
                encoding_state_as_density_matrix = DensityMatrix(self.input_size)
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
        if len(encoder) != self.input_size:
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

    def get_input_size(self) -> int:
        """Returns the number of input qubits"""
        return self.input_size
