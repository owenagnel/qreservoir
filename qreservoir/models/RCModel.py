import numpy as np
from qulacs import DensityMatrix, Observable
from qulacs.state import partial_trace
from qreservoir.reservoirs.Reservoir import Reservoir
from typing import List, Optional
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from typing import Any, Tuple


class RCModel:
    """Quantum reservoir computing model.

    This class describes a general quantum reservoir computing model that
    takes a reservoir, observable list, and a subestimator and fist and
    predicts sequential data. Note that multivariate time series are supported."""

    def __init__(
        self,
        reservoir: Reservoir,
        observables: List[Observable],
        subestimator: BaseEstimator,
        initial_state: Optional[DensityMatrix] = None,
    ) -> None:
        self.reservoir = reservoir
        self.observables = observables
        self.subestimator = subestimator
        if initial_state:
            self.initial_state = initial_state
        else:
            self.initial_state = DensityMatrix(self.reservoir.get_ancilla_num())
            self.initial_state.set_zero_state()

    def calculate_observables_for_sequence(
        self, input_seq: NDArray[np.double]
    ) -> NDArray[np.double]:
        """Takes a (n_samples, n_features) dimensional input sequence and returns an
        (n_samples, n_observables) array of observable values"""

        results = []
        prev = self.initial_state

        for sequent in input_seq:
            expections, prev = self.calculate_next_observable_list(sequent, prev)
            results.append(expections)

        return np.array(results)

    def calculate_next_observable_list(
        self, sequent: NDArray[np.double], prev: DensityMatrix
    ) -> Tuple[NDArray[np.double], DensityMatrix]:
        """Calculates the next state of the reservoir and the list of observable values"""

        # calculate the next state of the reservoir
        output_state = self.reservoir.get_reservoir_state(sequent, prev)
        # calculate the list of observable values
        expections = np.array(
            [ob.get_expectation_value(output_state) for ob in self.observables]
        )
        # trace out encoding qubit
        if self.reservoir.get_ancilla_num():
            prev = partial_trace(
                output_state,
                list(range(self.reservoir.get_encoding_qubit_num())),
            )
        return expections, prev

    def fit(
        self, X: NDArray[np.double]
    ) -> (
        None
    ):  # TODO, maybe allow for multiple sequence to be input, for more training data
        """X is an array_like of (n_samples, n_features), fit trains a model to
        predict the next value of the time series."""

        check_array(X)
        y = X[1:]
        X = X[:-1]
        prepared_data = self.calculate_observables_for_sequence(X)
        check_X_y(prepared_data, y, multi_output=True)
        self.subestimator.fit(prepared_data, y)
        self.is_fitted_ = True

    def predict(
        self, X: NDArray[np.double], additional_samples: int
    ) -> NDArray[np.double]:
        """Takes an input sequence X and returns the predicted next values of the sequence.
        'additional_samples' is the numer of additional datapoints the model will predict.
        (i.e. if additional_samples is 0, the model will return an array of prediction the
        same size as the input sequence)"""

        check_array(X)
        check_is_fitted(self, "is_fitted_")
        X_queue = list(X)
        outputs = []
        prev = self.initial_state
        for x in X_queue:
            expections, prev = self.calculate_next_observable_list(x, prev)
            # we must wrap expectations in a list because predict expects a 2d array,
            # and is not meant for time series predictions. Moreover, we must then
            # extract the first element out of the list because our subestimator is
            # returning 2d arrays in this instance
            output = self.subestimator.predict([expections])[0]
            if additional_samples:
                X_queue.append(output)
                additional_samples -= 1
            outputs.append(output)
        return np.array(outputs)

    def score(self, X: NDArray[np.double], y: NDArray[np.double]) -> np.double:
        """Returns the R^2 score of the model on the input sequence X and
        target sequence y"""

        additional_samples = len(y) - len(X)
        check_X_y(X, y, multi_output=True)
        y_pred = self.predict(X, additional_samples)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v
