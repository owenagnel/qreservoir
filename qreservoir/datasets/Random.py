import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from qreservoir.datasets.Dataset import Dataset, TrainTestSplit
from typing import Tuple
from numpy.typing import NDArray


class Random(Dataset):
    def __init__(self, size: int = 200) -> None:
        self.X, self.y = self.generate_data(size=size)

    def generate_data(self, size: int) -> Tuple[NDArray[np.double], NDArray[np.double]]:
        xs = np.reshape(np.linspace(0, np.pi, size), (size, 1))
        ys = np.random.uniform(0, 1, (size, 1))
        print(xs.shape, ys.shape)
        return xs, ys

    def get_train_test(self, test_size: float = 0.3) -> TrainTestSplit:
        return train_test_split(self.X, self.y, test_size=test_size, random_state=42)

    def show(self) -> None:
        x_min, x_max = self.X.min() - 0.5, self.X.max() + 0.5
        y_min, y_max = self.y.min() - 0.5, self.y.max() + 0.5

        # just plot the dataset first

        # Plot the training points
        plt.scatter(self.X[:, 0], self.y[:, 0], edgecolors="k")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()


if __name__ == "__main__":
    d = Random()
    d.show()
