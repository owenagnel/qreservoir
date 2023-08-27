from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray
from qulacs import DensityMatrix, QuantumCircuit, QuantumState


class Encoder(ABC):
    """Base class for encoders."""

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


class Reservoir(ABC):
    """Base class for reservoirs."""

    @abstractmethod
    def get_dynamics_circuit(self) -> QuantumCircuit:
        ...

    @abstractmethod
    def get_ancilla_num(self) -> int:
        ...

    @abstractmethod
    def get_encoding_qubit_num(self) -> int:
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


TrainTestSplit = Tuple[
    NDArray[np.double], NDArray[np.double], NDArray[np.double], NDArray[np.double]
]
"""
@private
Type alias for a train test split."""


class Dataset(ABC):
    """Base class for datasets."""

    @abstractmethod
    def get_train_test(self, test_size: float) -> TrainTestSplit:
        ...
