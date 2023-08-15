from abc import abstractmethod, ABC
from typing import Tuple
from numpy.typing import NDArray
import numpy as np

TrainTestSplit = Tuple[
    NDArray[np.double], NDArray[np.double], NDArray[np.double], NDArray[np.double]
]


class Dataset(ABC):
    @abstractmethod
    def get_train_test(self, test_size: float) -> TrainTestSplit:
        ...

    @abstractmethod
    def show(self) -> None:
        ...
