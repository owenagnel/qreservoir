from abc import abstractmethod, ABC
from qulacs import QuantumCircuit, QuantumState
import numpy as np
from numpy.typing import NDArray


class Encoder(ABC):
    @abstractmethod
    def get_circuit(self, input_vect: NDArray[np.double]) -> QuantumCircuit:
        ...

    @abstractmethod
    def get_encoding_state(self, input_vect: NDArray[np.double]) -> QuantumState:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def get_feature_num(self) -> int:
        ...

    @abstractmethod
    def print_circuit(self) -> None:
        ...
