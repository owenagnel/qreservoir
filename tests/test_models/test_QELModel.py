from qreservoir.models.QELModel import QELModel
from qreservoir.reservoirs.HarrRandomReservoir import HarrRandomReservoir
from qreservoir.reservoirs.CNOTReservoir import CNOTReservoir
from qreservoir.encoders.HEEncoder import HEEncoder
from qulacs import Observable, QuantumState
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.exceptions import NotFittedError
import pytest


def test_QELModel_sizes_1dim() -> None:
    encoder = HEEncoder(1, 2)
    reservoir = HarrRandomReservoir(encoder, 0)
    observable = Observable(1)
    observable.add_operator(1.0, "Z 0")
    observables = [observable]
    subestimator = LinearRegression()
    model = QELModel(reservoir, observables, subestimator)
    X = np.zeros((10, 1))  # must be a 2d array
    y = np.zeros(10)  # must be a 1d array
    model.fit(X, y)
    X_test = np.zeros((30, 1))
    out = model.predict(X_test)
    assert out.shape == (30,)
    assert out == pytest.approx(np.zeros((30,)))


def test_QELModel_sizes_2dim() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = HarrRandomReservoir(encoder, 0)
    observable = Observable(2)
    observable.add_operator(1.0, "Z 0")
    observables = [observable]
    subestimator = LinearRegression()
    model = QELModel(reservoir, observables, subestimator)
    X = np.zeros((10, 2))
    y = np.zeros(10)
    model.fit(X, y)
    X_test = np.zeros((30, 2))
    out = model.predict(X_test)
    assert out.shape == (30,)
    assert out == pytest.approx(
        np.zeros(
            30,
        )
    )


def test_QELModel_sizes_2dim_2obs() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = HarrRandomReservoir(encoder, 0)
    observable1 = Observable(2)
    observable1.add_operator(1.0, "Z 0")
    observable2 = Observable(2)
    observable2.add_operator(1.0, "Z 1")
    observables = [observable1, observable2]
    subestimator = LinearRegression()
    model = QELModel(reservoir, observables, subestimator)
    X = np.zeros((10, 2))
    y = np.zeros(10)
    model.fit(X, y)
    X_test = np.zeros((30, 2))
    out = model.predict(X_test)
    assert out.shape == (30,)
    assert out == pytest.approx(
        np.zeros(
            30,
        )
    )


def test_initial_state_works() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = CNOTReservoir(
        encoder, 3, 0
    )  # NB depth is zero so no dynamics take place
    observable = Observable(5)
    observable.add_operator(1.0, "Z 2")
    observable.add_operator(1.0, "Z 3")
    observable.add_operator(1.0, "Z 4")
    observables = [observable]
    subestimator = LinearRegression()
    initial = QuantumState(3)
    initial.set_computational_basis(7)
    model = QELModel(reservoir, observables, subestimator, initial_state=initial)

    expect = model.calculate_observable_expectations(np.random.uniform(size=(2,)))
    assert expect[0] == pytest.approx(
        -3
    )  # encoding state should still be |0> so expectation of sum of paulis should be 2


def test_can_predict_constant_data() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = HarrRandomReservoir(encoder, 3)
    observable = Observable(2)
    observable.add_operator(1.0, "Z 0")
    observables = [observable]
    subestimator = LinearRegression()
    model = QELModel(reservoir, observables, subestimator)
    X = np.random.uniform(-np.pi, np.pi, size=(10, 2))
    y = np.ones(10) * 4
    model.fit(X, y)
    X_test = np.random.uniform(-np.pi, np.pi, size=(30, 2))
    out = model.predict(X_test)
    assert out.shape == (30,)
    assert out == pytest.approx(np.ones(shape=(30,)) * 4)


def test_raises_NotFittedError_if_not_fitted() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = HarrRandomReservoir(encoder, 3)
    observable = Observable(2)
    observable.add_operator(1.0, "Z 0")
    observables = [observable]
    subestimator = LinearRegression()
    model = QELModel(reservoir, observables, subestimator)
    X = np.ones((10, 2))
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_observable_calculation() -> None:
    encoder = HEEncoder(1, 0)
    reservoir = CNOTReservoir(encoder, 1, 1)
    observable_anc = Observable(2)
    initial = QuantumState(1)
    initial.set_computational_basis(1)
    observable_anc.add_operator(
        1.0, "Z 1"
    )  # this measures the ancilla qubit (reversed numbering standard in qulcas)
    observable_enc = Observable(2)
    observable_enc.add_operator(1.0, "Z 0")
    observables = [observable_anc, observable_enc]
    model = QELModel(reservoir, observables, LinearRegression(), initial_state=initial)

    expectations = model.calculate_observable_expectations(
        np.random.uniform(-np.pi, np.pi, (1))
    )  # the input data doesn't matter since encoder has depth 0
    assert expectations.shape == (2,)
    assert expectations[0] == pytest.approx(-1)  # anc
    assert expectations[1] == pytest.approx(-1)  # enc


def test_scoring_peformance() -> None:
    ...  # TODO write scoring tests


def test_raises_error_incorrectly_shaped_X_fit() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = HarrRandomReservoir(encoder, 0)
    observable1 = Observable(2)
    observable1.add_operator(1.0, "Z 0")
    observable2 = Observable(2)
    observable2.add_operator(1.0, "Z 1")
    observables = [observable1, observable2]
    subestimator = LinearRegression()
    model = QELModel(reservoir, observables, subestimator)
    X = np.zeros(10)  # inccorect shape 1d
    with pytest.raises(ValueError):
        model.fit(X, np.zeros(10))

    X = np.zeros((10, 3, 4))  # incorrect shape 3d
    with pytest.raises(ValueError):
        model.fit(X, np.zeros(10))

    X = np.zeros((10, 3))  # incorrect feature num
    with pytest.raises(ValueError):
        model.fit(X, np.zeros(10))

    X = np.zeros((10, 1))  # incorrect feature num
    with pytest.raises(ValueError):
        model.fit(X, np.zeros(10))


def test_raises_error_incorrectly_shaped_X_predict() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = HarrRandomReservoir(encoder, 0)
    observable1 = Observable(2)
    observable1.add_operator(1.0, "Z 0")
    observable2 = Observable(2)
    observable2.add_operator(1.0, "Z 1")
    observables = [observable1, observable2]
    subestimator = LinearRegression()
    model = QELModel(reservoir, observables, subestimator)
    X = np.zeros((10, 2))
    model.fit(X, np.zeros(10))
    with pytest.raises(ValueError):
        model.predict(np.zeros((10, 2, 1)))
    with pytest.raises(ValueError):
        model.predict(np.zeros(10))


def test_raises_error_incorrectly_shaped_y_fit() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = HarrRandomReservoir(encoder, 0)
    observable1 = Observable(2)
    observable1.add_operator(1.0, "Z 0")
    observable2 = Observable(2)
    observable2.add_operator(1.0, "Z 1")
    observables = [observable1, observable2]
    subestimator = LinearRegression()
    model = QELModel(reservoir, observables, subestimator)
    X = np.zeros((10, 2))
    with pytest.raises(ValueError):
        model.fit(X, np.zeros((10, 2)))
