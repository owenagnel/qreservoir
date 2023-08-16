from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from qreservoir.datasets.Dataset import Dataset, TrainTestSplit
from typing import Tuple
from numpy.typing import NDArray


class MackeyGlass(Dataset):
    def __init__(self, size: int = 200) -> None:
        """Warning: as this is a time series, y has shape (size, 1), rather than (size, ).
        When passing things into sklearn, be careful of shapes of arrays."""
        self.size = size
        self.b = 0.05
        self.c = 0.15
        self.tau = 20
        self.X, self.y = self.generate_data()

    def generate_data(self) -> Tuple[NDArray[np.double], NDArray[np.double]]:
        start_val_num = 1 + self.tau
        y = list(np.random.uniform(0.95, 1.05, start_val_num))
        for n in range(start_val_num - 1, self.size + 99):
            y.append(
                y[n]
                - (
                    self.b * y[n]
                    + self.c * y[n - self.tau] / (1 + y[n - self.tau] ** n)
                )
            )
        y = y[100:]
        return np.linspace(0, 10, self.size), np.reshape(np.array(y), (self.size, 1))

    def get_train_test(self, test_size: float = 0.3) -> TrainTestSplit:
        return train_test_split(self.X, self.y, test_size=test_size, random_state=42)

    def show(self) -> None:
        x_min, x_max = self.X.min() - 0.5, self.X.max() + 0.5
        y_min, y_max = self.y.min() - 0.5, self.y.max() + 0.5

        # just plot the dataset first

        # Plot the training points
        plt.scatter(self.X, self.y[:, 0], edgecolors="k")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()


if __name__ == "__main__":
    d = MackeyGlass(size=1000)
    d.show()
