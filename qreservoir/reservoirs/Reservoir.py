from abc import abstractmethod
from qulacs import QuantumCircuit, QuantumState, DensityMatrix
import numpy as np
from typing import Union
from numpy.typing import NDArray
from qreservoir.encoders.Encoder import Encoder


class Reservoir:
    @abstractmethod
    def get_dynamics_circuit(self) -> QuantumCircuit:
        ...

    @abstractmethod
    def get_ancilla_num(self) -> int:
        ...

    @abstractmethod
    def get_input_size(self) -> int:
        ...

    @abstractmethod
    def get_reservoir_state(
        self,
        input_vect: NDArray[np.double],
        prev: Union[QuantumState, DensityMatrix, None],
    ) -> Union[QuantumState, DensityMatrix]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def connect_encoder(self, encoder: Encoder) -> None:
        ...

    @abstractmethod
    def print_circuit(self) -> None:
        ...
