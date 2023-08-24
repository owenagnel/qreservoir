import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from qreservoir.datasets.Dataset import Dataset, TrainTestSplit
from typing import Tuple
from numpy.typing import NDArray


class Complex_Fourrier(Dataset):
    def __init__(
        self, noise: float = 0.5, size: int = 200, complexity: int = 5
    ) -> None:
        "Generates a fourrier series with random frequency coefficients and `complexity` terms"
        self.random_coef = np.random.uniform(-1, 1, (complexity, 2))
        self.X, self.y = self.generate_data(
            noise=noise, size=size, complexity=complexity
        )

    def generate_data(
        self, noise: float, size: int, complexity: int
    ) -> Tuple[NDArray[np.double], NDArray[np.double]]:
        xs = np.reshape(np.linspace(0, np.pi, size), (size, 1))
        ys = np.zeros(size)
        for i in range(complexity):
            ys = np.add(self.random_coef[i][0] * np.sin(i * xs[:, 0]), ys)
            ys = np.add(self.random_coef[i][1] * np.cos(i * xs[:, 0]), ys)
        ys += noise * 0.1 * np.random.normal(-1, 1, (size))
        return xs, np.reshape(ys, (size,))

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
    d = Complex_Fourrier(noise=0.0, size=300, complexity=5)
    d.show()
