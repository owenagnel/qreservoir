from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split

from qreservoir.abstract_base_classes import Dataset, TrainTestSplit


class Circles(Dataset):
    """A circles classification dataset. `X` is an array of shape `(size, 2)`
    and `y` is an array of shape `(size,)` containing labels. Returns the scikit-learn
    make_circles dataset with the passed parameters."""

    noise: float
    """
    @private
    The noise parameter passed to scikit-learn make_circles."""

    size: int
    """
    @private
    The size parameter is the number of data points to generate."""

    def __init__(self, noise: float = 0.5, size: int = 200) -> None:
        r"""Initialises the dataset object with passed parameters.

        Parameters
        ----------
        size : float, optional
            The number of data points to generate. The default is 200.
        noise : float, optional
            The noise passed to scikit-learn make_circles. The default is 0.5.
        """
        self.noise = noise
        self.size = size

    def get_train_test(self, test_size: float = 0.3) -> TrainTestSplit:
        """Returns a train test split of the data."""
        X, y = make_circles(noise=self.noise, factor=0.3, n_samples=self.size)
        return train_test_split(X, y, test_size=test_size, random_state=42)


class LinearlySeperable(Dataset):
    """A linearly seperable classification dataset. `X` is an array of shape `(size, 2)`
    and `y` is an array of shape `(size,)` containing labels. Returns the scikit-learn
    make_classification dataset with the passed parameters. Noise is added to the dataset via
    a multivariqte gaussian."""

    noise: float
    """
    @private
    The magnitude of noise we add."""

    size: int
    """
    @private
    The size parameter is the number of data points to generate."""

    def __init__(self, noise: float = 0.5, size: int = 200) -> None:
        r"""Initialises the dataset object with passed parameters.

        Parameters
        ----------
        size : float, optional
            The number of data points to generate. The default is 200.
        noise : float, optional
            The noise factor we multiply the gaussian noise by. The default is 0.5.
        """
        self.noise = noise
        self.size = size

    def get_train_test(self, test_size: float = 0.3) -> TrainTestSplit:
        """Returns a train test split of the data."""
        X, y = make_classification(
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            n_samples=self.size,
        )
        rng = np.random.RandomState()
        X += 4 * self.noise * rng.uniform(size=X.shape)
        return train_test_split(X, y, test_size=test_size)


class Moons(Dataset):
    """A moons classification dataset. `X` is an array of shape `(size, 2)`
    and `y` is an array of shape `(size,)` containing labels.
    Returns the scikit-learn make_moons dataset with the passed parameters."""

    noise: float
    """
    @private
    The noise parameter passed to scikit-learn make_moons."""

    size: int
    """
    @private
    The size parameter is the number of data points to generate."""

    def __init__(self, noise: float = 0.2, size: int = 200) -> None:
        r"""Initialises the dataset object with passed parameters.

        Parameters
        ----------
        size : float, optional
            The number of data points to generate. The default is 200.
        noise : float, optional
            The noise passed to scikit-learn make_moons. The default is 0.2.
        """
        self.noise = noise
        self.size = size

    def get_train_test(self, test_size: float = 0.3) -> TrainTestSplit:
        X, y = make_moons(noise=self.noise, n_samples=self.size)
        return train_test_split(X, y, test_size=test_size)


class Complex_Fourrier(Dataset):
    r"""A truncated random fourrier series regression dataset. `X` is 2D arrays of shape `(size, 1)` and `y` os a 1D array of shape `(size,)`.
    `X` is a linspace between 0 and :math:`2 \pi`, and `y` is :math:`f(x)` for a given :math:`x`.

    .. math:: f(x) = \sum_{r=0}^{N} a_r \sin(r x) + b_r \cos(r x)

    Where :math:`\mathbf{a}` and :math:`\mathbf{b}` are vectors of uniformly distributed random coefficients between -1 and 1 and
    :math:`N` is `complexity`.
    """

    noise: float
    """
    @private
    The noise parameter is the standard deviation of the gaussian noise added to the fourrier series."""

    size: int
    """
    @private
    The size parameter is the number of data points to generate."""

    complexity: int
    """
    @private
    The complexity parameter is the number of cos and sin terms in the fourrier series."""

    seed: Optional[int]
    """
    @private
    The seed parameter makes data generation deterministic. If None, the seed is not set."""

    def __init__(
        self,
        noise: float = 0.5,
        size: int = 200,
        complexity: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        r"""Initialises the dataset object with passed parameters.

        Parameters
        ----------
        size : float, optional
            The number of data points to generate. The default is 200.
        noise : float, optional
            The standard deviation of noise to add to the fourrier series. The default is 0.5.
        complexity : int, optional
            The number of terms in the fourrier series. The default is 5.

        Other Parameters
        ----------------
        seed : int, optional
            Seed parameter to make data generation deterministic. The default is None.
        """
        self.noise = noise
        self.size = size
        self.complexity = complexity
        self.seed = seed

    def generate_data(self) -> Tuple[NDArray[np.double], NDArray[np.double]]:
        """
        @private
        Generates a truncated random fourrier series."""
        if self.seed:
            np.random.seed(self.seed)
        self.random_coef = np.random.uniform(-1, 1, (self.complexity, 2))
        xs = np.reshape(np.linspace(0, 2 * np.pi, self.size), (self.size, 1))
        ys = np.zeros(self.size)
        for i in range(self.complexity):
            ys = np.add(self.random_coef[i][0] * np.sin((i + 1) * xs[:, 0]), ys)
            ys = np.add(self.random_coef[i][1] * np.cos((i + 1) * xs[:, 0]), ys)
        ys += np.random.normal(0, self.noise, (self.size))
        np.random.seed()  # rerandomise seed
        return xs, ys

    def get_train_test(self, test_size: float = 0.3) -> TrainTestSplit:
        """Returns a train test split of the data."""
        X, y = self.generate_data()
        return train_test_split(X, y, test_size=test_size, random_state=42)


class MackeyGlass(Dataset):
    """A MackeyGlass time series regression dataset. `X` is 2D arrays of shape `(size, 1)` and `y` os a 1D array of shape `(size,)`.
    `X` is a linspace between 0 and 10, and `y` is the corresponding MackeyGlass value at that time step
    """

    size: int
    """
    @private
    The size parameter is the number of data points to generate."""

    b: float
    """
    @private
    The b parameter is a constant in the MackeyGlass equation."""

    c: float
    """
    @private
    The c parameter is a constant in the MackeyGlass equation."""

    tau: int
    """
    @private
    The tau parameter is a constant in the MackeyGlass equation. Depth of history."""

    def __init__(
        self, size: int = 200, b: float = 0.05, c: float = 0.15, tau: int = 20
    ) -> None:
        r"""Initialises the dataset object with passed parameters.

        Parameters
        ----------
        size : float, optional
            The number of data points to generate. The default is 200.

        Other Parameters
        ----------------
        b : float, optional
            The b parameter is a constant in the MackeyGlass equation. The default is 0.05.
        c : float, optional
            The c parameter is a constant in the MackeyGlass equation. The default is 0.15.
        tau : int, optional
            The tau parameter is a constant in the MackeyGlass equation. Depth of history. The default is 20.
        """
        self.size = size
        self.b = b
        self.c = c
        self.tau = tau

    def generate_data(self) -> Tuple[NDArray[np.double], NDArray[np.double]]:
        """
        @private
        Generates a MackeyGlass time series."""
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
        return np.reshape(np.linspace(0, 10, self.size), (self.size, 1)), np.array(y)

    def get_train_test(self, test_size: float = 0.3) -> TrainTestSplit:
        """Returns a train test split of the data."""
        X, y = self.generate_data()
        return train_test_split(X, y, test_size=test_size, random_state=42)


class Random(Dataset):
    """A random regression dataset of uniformly uniform distribution between 0 and 1. `X` is 2D arrays of shape `(size, 1)` and `y` os a 1D array of shape `(size,)`.
    2D arrays of shape `(size, 1)`. `X` is a linspace between 0 and :math:`2 \pi`, and `y` is a random uniform.
    """

    seed: Optional[int]
    """
    @private
    The seed parameter makes data generation deterministic. If None, the seed is not set."""

    size: int
    """
    @private
    The size parameter is the number of data points to generate."""

    def __init__(self, size: int = 200, seed: Optional[int] = None) -> None:
        """Initialises the dataset object with passed parameters.

        Parameters
        ----------
        size : int, optional
            The number of data points to generate. The default is 200.

        Other Parameters
        ----------------
        seed : int, optional
            Seed parameter to make data generation deterministic."""
        self.seed = seed
        self.size = size

    def generate_data(self) -> Tuple[NDArray[np.double], NDArray[np.double]]:
        """
        @private
        Generates a random dataset of uniformly uniform distribution between 0 and 1."""
        if self.seed:
            np.random.seed(self.seed)
        xs = np.reshape(np.linspace(0, 2 * np.pi, self.size), (self.size, 1))
        ys = np.random.uniform(0, 1, (self.size,))
        np.random.seed()  # rerandomise seed
        return xs, ys

    def get_train_test(self, test_size: float = 0.3) -> TrainTestSplit:
        """Returns a train test split of the data."""
        X, y = self.generate_data()
        return train_test_split(X, y, test_size=test_size, random_state=42)


class Sine(Dataset):
    """The sine regression dataset is a sine wave with gaussian noise added. `X` is a linspace between 0 and :math:`2 \pi`,
    and `y` is the corresponding sine wave with gaussian noise added. `X` is 2D arrays of shape `(size, 1)` and `y` os a 1D array of shape `(size,)`.
    """

    noise: float
    """
    @private
    The noise parameter is the standard deviation of the gaussian noise added to the sine wave."""

    size: int
    """
    @private
    The size parameter is the number of data points to generate."""

    def __init__(self, size: int = 200, noise: float = 0.5) -> None:
        r"""Initialises the dataset object with passed paramters.

        Parameters
        ----------
        size : int, optional
            The number of data points to generate. The default is 200.
        noise : float, optional
            The magintude of noise to add to the sine wave. The default is 0.5.

        """
        self.noise = noise
        self.size = size

    def generate_data(self) -> Tuple[NDArray[np.double], NDArray[np.double]]:
        """
        @private
        Generates a sine wave with gaussian noise added."""
        xs = np.reshape(np.linspace(0, 2 * np.pi, self.size), (self.size, 1))
        ys = np.sin(xs[:, 0])
        ys += np.random.normal(0.0, self.noise, (self.size))
        return xs, ys

    def get_train_test(self, test_size: float = 0.3) -> TrainTestSplit:
        """Returns a train test split of the data."""
        X, y = self.generate_data()
        return train_test_split(X, y, test_size=test_size, random_state=42)
