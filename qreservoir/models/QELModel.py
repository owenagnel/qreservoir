import numpy as np
from numpy.typing import NDArray
from typing import List, Optional
from qulacs import Observable, QuantumState
from qreservoir.reservoirs.Reservoir import Reservoir
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class QELModel:
    """General quantum extreme learning model.

    This class describes a general quantum extreme learning model that
    takes a reservoir, list of observables, and subestimator and fits
    and predicts data. Note that the target values must be scalars."""

    def __init__(
        self,
        reservoir: Reservoir,
        observables: List[Observable],
        subestimator: BaseEstimator,
        initial_state: Optional[QuantumState] = None,
    ) -> None:
        self.reservoir = reservoir
        self.observables = observables
        self.subestimator = subestimator
        if initial_state:
            self.initial_state = initial_state
        else:
            self.initial_state = QuantumState(self.reservoir.get_ancilla_num())
            self.initial_state.set_zero_state()

    def calculate_observable_expectations(
        self, input_vec: NDArray[np.double]
    ) -> NDArray[np.double]:
        """Calculates the list of observable values after dynamics given
        an array_like input of size (n_features) and initial state. Returns
        1d array_like of size (n_observables)"""

        # calculate the next state of the reservoir
        output_state = self.reservoir.get_reservoir_state(input_vec, self.initial_state)
        # calculate the list of observable values
        expections = np.array(
            [ob.get_expectation_value(output_state) for ob in self.observables]
        )

        return expections

    def batch_calculate_observable_expectations(
        self, input_vecs: NDArray[np.double]
    ) -> NDArray[np.double]:
        """Calculates observable values for an array_like batch of input vectors
        of size (n_samples, n_features)."""

        outputs = []
        for input_vec in input_vecs:
            outputs.append(self.calculate_observable_expectations(input_vec))
        return np.array(outputs)

    def fit(self, X: NDArray[np.double], y: NDArray[np.double]) -> None:
        """X is an array_like of (n_samples, n_features), fit trains a model to
        predict a new output, note y must be a 1_d array_like with shape (n_samples,)"""

        check_array(X)
        prepared_data = self.batch_calculate_observable_expectations(X)
        check_X_y(prepared_data, y)
        self.subestimator.fit(prepared_data, y)
        self.is_fitted_ = True

    def predict(self, X: NDArray[np.double]) -> NDArray[np.double]:
        """X is an array_like of (n_samples, n_features), returns the predicted
        output shape (n_samples,)"""

        check_array(X)
        check_is_fitted(self, "is_fitted_")
        expections = self.batch_calculate_observable_expectations(X)
        return self.subestimator.predict(expections)

    def score(self, X: NDArray[np.double], y: NDArray[np.double]) -> np.double:
        """Returns the R^2 score of the model on the input sequence
        X and target sequence y"""

        check_X_y(X, y)
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v
