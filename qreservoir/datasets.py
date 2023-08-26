from abc import abstractmethod, ABC
from typing import Tuple, Optional
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles, make_classification, make_moons


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


class Circles(Dataset):
    def __init__(self, noise: float = 0.5, size: int = 200) -> None:
        self.X, self.y = make_circles(noise=noise, factor=0.3, n_samples=size)

    def get_train_test(self, test_size: float = 0.3) -> TrainTestSplit:
        return train_test_split(self.X, self.y, test_size=test_size, random_state=42)

    def show(self) -> None:
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5

        # just plot the dataset first
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        plt.title("Input data")

        # Plot the training points
        plt.scatter(
            self.X[:, 0], self.X[:, 1], c=self.y, cmap=cm_bright, edgecolors="k"
        )

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()


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


class LinearlySeperable(Dataset):
    def __init__(self, noise: float = 0.5, size: int = 200) -> None:
        self.X, self.y = make_classification(
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            n_samples=size,
        )
        rng = np.random.RandomState(2)
        self.X += 4 * noise * rng.uniform(size=self.X.shape)
        self.linearly_separable = (self.X, self.y)

    def get_train_test(self, test_size: float = 0.3) -> TrainTestSplit:
        return train_test_split(self.X, self.y, test_size=test_size, random_state=42)

    def show(self) -> None:
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5

        # just plot the dataset first
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        plt.title("Input data")

        # Plot the training points
        plt.scatter(
            self.X[:, 0], self.X[:, 1], c=self.y, cmap=cm_bright, edgecolors="k"
        )

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()


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


class Moons(Dataset):
    def __init__(self, noise: float = 0.2, size: int = 200) -> None:
        self.X, self.y = make_moons(noise=noise, n_samples=size)

    def get_train_test(self, test_size: float = 0.3) -> TrainTestSplit:
        return train_test_split(self.X, self.y, test_size=test_size, random_state=42)

    def show(self) -> None:
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5

        # just plot the dataset first
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        plt.title("Input data")

        # Plot the training points
        plt.scatter(
            self.X[:, 0], self.X[:, 1], c=self.y, cmap=cm_bright, edgecolors="k"
        )

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()


class Random(Dataset):
    def __init__(self, size: int = 200, seed: Optional[int] = None) -> None:
        if seed:
            np.random.seed(seed)
        self.X, self.y = self.generate_data(size=size)
        np.random.seed()

    def generate_data(self, size: int) -> Tuple[NDArray[np.double], NDArray[np.double]]:
        xs = np.reshape(np.linspace(0, np.pi, size), (size, 1))
        ys = np.random.uniform(0, 1, (size, 1))
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


class Sine(Dataset):
    def __init__(self, noise: float = 0.5, size: int = 200) -> None:
        self.X, self.y = self.generate_data(noise=noise, size=size)

    def generate_data(
        self, noise: float = 0.5, size: int = 200
    ) -> Tuple[NDArray[np.double], NDArray[np.double]]:
        xs = np.reshape(np.linspace(0, np.pi, size), (size, 1))
        ys = np.sin(xs[:, 0])
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
