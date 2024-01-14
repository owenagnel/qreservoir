from typing import List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from qulacs import DensityMatrix, Observable, QuantumState
from qulacs.state import partial_trace
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from qreservoir.reservoirs import Reservoir

OBSERVABLE_SETS = ["Total-Z", "IC-POVM"]


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
        samples: int = 1,
    ) -> None:
        r"""Initialises the QELM model given a reservoir and estimator.

        Parameters
        ----------
        reservoir : Reservoir
            The reservoir, encoder pair to be used by the model.
        observables : List[Observable]
            A list of user defined qulacs 'Observable` objects.
        subestimator : BaseEstimator
            A `scikit-learn` estimator to fit the data once processed by the reservoir.
        sample : int
            If we have noise in the system, number of samples for estimating expectation values

        Other Parameters
        ----------------
        initial_state : QuantumState, optional
            The initial state of the reservoir hidden space, by default None. Must be the same size as the reservoir ancillas.
        """
        self.reservoir = reservoir
        self.observables = observables
        self.subestimator = subestimator
        self.samples = samples
        if initial_state:
            if initial_state.get_qubit_count() != self.reservoir.get_ancilla_num():
                raise ValueError(
                    "initial_state qubit count and reservoir ancilla qubit count don't match"
                )
            self.initial_state = initial_state
        else:
            self.initial_state = QuantumState(self.reservoir.get_ancilla_num())
            self.initial_state.set_zero_state()

    def calculate_observable_expectations(
        self, input_vec: NDArray[np.double]
    ) -> NDArray[np.double]:
        r"""Calculates the list of observable values after dynamics given
        an array of inputs and predefined initial state. Returns the list of observable expectations.

        Parameters
        ----------
        input_vec : NDArray[np.double]
            A 1d array of input values of shape `(encoder.feature_num,)` to be passed to the reservoir.

        Returns
        -------
        NDArray[np.double]
            A 1d array of observable expectations with shape `(len(observables),)` after encoding and dynamics.
        """
        # calculate the next state of the reservoir
        # output_state = self.reservoir.get_reservoir_state(input_vec, self.initial_state)
        # calculate the list of observable values
        expectations = np.array(
            [
                [
                    ob.get_expectation_value(
                        self.reservoir.get_reservoir_state(
                            input_vec, self.initial_state
                        )
                    )
                    for ob in self.observables
                ]
                for _ in range(self.samples)
            ]
        )
        expectations = np.mean(expectations, axis=0)

        return expectations

    def batch_calculate_observable_expectations(
        self, input_vecs: NDArray[np.double]
    ) -> NDArray[np.double]:
        r"""Calculates a batch of observable expectation values after dynamics given an batch of inputs
        and a predefined initial state. Returns the list of observable expectations.

        Parameters
        ----------
        input_vecs : NDArray[np.double]
            A 2d array of input values of shape `(n_samples, encoder.feature_num)` to be passed to the reservoir.

        Returns
        -------
        NDArray[np.double]
            A 2d array of observable expectations with shape `(n_samples, len(observables))` after encoding and dynamics.
        """

        outputs = []
        for input_vec in input_vecs:
            outputs.append(self.calculate_observable_expectations(input_vec))
        return np.array(outputs)

    def fit(self, X: NDArray[np.double], y: NDArray[np.double]) -> None:
        r"""Takes a training dataset and fits the QELM model using the subestimator passed at creation

        Parameters
        ----------
        X : NDArray[np.double]
            A 2d array of input values of shape `(n_samples, encoder.feature_num)`.

        y : NDArray[np.double]
            A 1d array of target values of shape `(n_samples,)`. Note that `scikit-learn` allows for multivariate outputs, so a
            y with shape `(n_samples, n_outputs)` is also valid. However, the method will raise a warning and this approach is not in general recommended.
        """

        check_array(X)
        prepared_data = self.batch_calculate_observable_expectations(X)
        check_X_y(prepared_data, y)
        self.subestimator.fit(prepared_data, y)
        self.is_fitted_ = True

    def predict(self, X: NDArray[np.double]) -> NDArray[np.double]:
        r"""Returns the output values predicted by the QELM model.

        Parameters
        ----------
        X : NDArray[np.double]
            A 2d array of input values of shape `(n_samples, encoder.feature_num)`.

        Returns
        -------
        NDArray[np.double]
            A 1d array of predicted values of shape `(n_samples,)`. (note that if the model was
            fitted with mutlivariate outputs, the shape will be `(n_samples, n_outputs)`)
        """

        check_array(X)
        check_is_fitted(self, "is_fitted_")
        expections = self.batch_calculate_observable_expectations(X)
        return self.subestimator.predict(expections)

    def score(self, X: NDArray[np.double], y: NDArray[np.double]) -> np.double:
        r"""Returns R^2 score of the model on the test data.

        Parameters
        ----------
        X : NDArray[np.double]
            A 2d array of input values of shape `(n_samples, encoder.feature_num)`.

        y : NDArray[np.double]
            A 1d array of target values of shape `(n_samples,)`. Note that `scikit-learn` allows for multivariate outputs, so a
            y with shape `(n_samples, n_outputs)` is also valid. However, the method will raise a warning and this approach is not in general recommended.

        Returns
        -------
        np.double
            The R^2 score of the model on the test data.
        """

        check_X_y(X, y)
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v


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
        r"""Initialises the QELM model given a reservoir and estimator.

        Parameters
        ----------
        reservoir : Reservoir
            The reservoir, encoder pair to be used by the model.
        observables : List[Observable]
            A list of user defined qulacs 'Observable` objects.
        subestimator : BaseEstimator
            A `scikit-learn` estimator to fit the data once processed by the reservoir.

        Other Parameters
        ----------------
        initial_state : QuantumState, optional
            The initial state of the reservoir hidden space, by default None. Must be the same size as the reservoir ancillas.
        """
        self.reservoir = reservoir
        self.observables = observables
        self.subestimator = subestimator
        if initial_state:
            self.initial_state = initial_state
        else:
            self.initial_state = DensityMatrix(self.reservoir.get_ancilla_num())
            self.initial_state.set_zero_state()

    def calculate_observables_for_sequence(
        self, input_seq: NDArray[np.double], index: Optional[int] = None
    ) -> NDArray[np.double]:
        r"""Takes a time series input sequence and returns an array of observable values expectations
        for each time step. If index is specified, only the observable values for the indexth
        sample will be returned.

        Parameters
        ----------
        input_seq : NDArray[np.double]
            A 2d array of input values of shape `(n_timesteps, encoder.feature_num)`.

        Other Parameters
        ----------------
        index : int, optional
            The index of the timestep to return the observable values for, by default None. If None, the observable values for all timesteps will be returned.

        Returns
        -------
        NDArray[np.double]
            A 2d array of observable expectations with shape `(n_timesteps, len(observables))` after encoding and dynamics. Note that if `index` is specified, the shape will be `(1, len(observables))`
        """

        if index and (index < 0 or index >= len(input_seq)):
            raise ValueError("index must be between 0 and len(input_seq) - 1")

        results = []
        prev = self.initial_state

        for step, sequent in enumerate(input_seq):
            if index is None:
                expections, prev = self.next_reservoir_state_with_observables(
                    prev, sequent
                )
                results.append(expections)
            elif step == index:
                expections, prev = self.next_reservoir_state_with_observables(
                    prev, sequent
                )
                results.append(expections)
                break  # all further reservoir states aren't needed
            else:
                prev = self.next_reservoir_state(prev, sequent)

        return np.array(results)

    def next_reservoir_state_with_observables(
        self, prev: DensityMatrix, sequent: NDArray[np.double]
    ) -> Tuple[NDArray[np.double], DensityMatrix]:
        r"""Takes a single timestep input (`sequent`) and the previous state of the reservoir and returns the next state of the reservoir and the list of observable values.

        Parameters
        ----------
        prev : DensityMatrix
            The state of the reservoir at the previous timestep. (Note that the dimension of this state must be that of teh hidden space of the reservoir)

        sequent : NDArray[np.double]
            A 1d array of input values of shape `(encoder.feature_num,)` to be passed to the reservoir.

        Returns
        -------
        NDArray[np.double]
            A 2d array of observable expectations with shape `(len(observables),)` after encoding and dynamics.

        DensityMatrix
            The state of the reservoir after dynamics with encoding qubits traceds out.
        """

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

    def next_reservoir_state(
        self, prev: DensityMatrix, sequent: NDArray[np.double]
    ) -> DensityMatrix:
        r"""Calculates only the next state of the reservoir without expectation values (used as helper function).

        Parameters
        ----------
        prev : DensityMatrix
            The state of the reservoir at the previous timestep. (Note that the dimension of this state must be that of teh hidden space of the reservoir)

        sequent : NDArray[np.double]
            A 1d array of input values of shape `(encoder.feature_num,)` to be passed to the reservoir.

        Returns
        -------
        DensityMatrix
            The state of the reservoir after dynamics with encoding qubits traceds out.
        """

        output_state = self.reservoir.get_reservoir_state(sequent, prev)
        if self.reservoir.get_ancilla_num():
            prev = partial_trace(
                output_state,
                list(range(self.reservoir.get_encoding_qubit_num())),
            )
        else:
            prev = cast(DensityMatrix, output_state)  # should never happen
        return prev

    def fit(
        self, X: NDArray[np.double]
    ) -> None:  # TODO, allow for multiple sequence to be inputfor more training data
        """X is an array_like of (n_samples, n_features), fit trains a model to
        predict the next value of the time series."""

        r"""Takes a training dataset and fits the QELM model using the subestimator passed at creation

        Parameters
        ----------
        X : NDArray[np.double]
            A 2d array of input values of shape `(n_timesteps, encoder.feature_num)`.

        Notes
        -----
        The target values are assumed to be the next value in the sequence. (i.e. if the input sequence is [[1],[2],[3],[4],[5]], the target sequence is [[2],[3],[4],[5],[6]]).
        This means that the scikit-learn estimator will return an array rather than a scalar value when asked to predict. Hence, the number of encoding features and predicted
        features are assumed to be identical. I.e. `encoder.feature_num` = `n_outputs`.

        This method should also be updated to take a batch of training data sequences rather than a single one (i.e. X should be a 3d array)
        """

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
        r"""Takes an input sequence X and returns the predicted next values of the sequence predicted by the model.

        Parameters
        ----------
        X : NDArray[np.double]
            A 2d array of input values of shape `(n_timesteps, encoder.feature_num)`.

        additional_samples : int
            The number of additional samples to predict after the input sequence. (note that this means
            if additional_samples is 0, the model will return an array of prediction the
            same size as the input sequence)

        Returns
        -------
        NDArray[np.double]
            A 2d array of predicted values of shape `(n_timesteps + additional_samples, encoder.feature_num)`.
        """

        check_array(X)
        check_is_fitted(self, "is_fitted_")
        X_queue = list(X)
        outputs = []
        prev = self.initial_state
        for x in X_queue:
            expections, prev = self.next_reservoir_state_with_observables(prev, x)
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
        r"""Returns R^2 score of the model on the test data. X is a truncated time series and y is the full sequences.

        Parameters
        ----------
        X : NDArray[np.double]
            A 2d array of input values of shape `(n_timesteps, encoder.feature_num)`.

        y : NDArray[np.double]
           A 2d array of input values of shape `(n_test_timesteps, encoder.feature_num)`. With n_test_timesteps > n_timesteps

        Returns
        -------
        np.double
            The R^2 score of the model on the test data.

        Raises
        ------
        ValueError
            If the length of y is less than the length of X.
        """

        additional_samples = len(y) - len(X)
        if additional_samples < 0:
            raise ValueError("y must be at least as long as X")

        check_X_y(X, y, multi_output=True)
        y_pred = self.predict(X, additional_samples)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v
