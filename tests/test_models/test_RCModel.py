from qreservoir.models.RCModel import RCModel
from qreservoir.reservoirs.HarrRandomReservoir import HarrRandomReservoir
from qreservoir.reservoirs.CNOTReservoir import CNOTReservoir
from qreservoir.encoders.HEEncoder import HEEncoder
from qulacs import Observable, DensityMatrix
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.exceptions import NotFittedError
import pytest


def test_RCModel_sizes_1dim() -> None:
    encoder = HEEncoder(1, 2)
    reservoir = HarrRandomReservoir(encoder, 0)
    observable = Observable(1)
    observable.add_operator(1.0, "Z 0")
    observables = [observable]
    subestimator = LinearRegression()
    model = RCModel(reservoir, observables, subestimator)
    X = np.zeros((10, 1))
    model.fit(X)
    X_test = np.zeros((30, 1))
    out = model.predict(X_test, 10)
    assert out.shape == (40, 1)
    assert out == pytest.approx(np.zeros((40, 1)))


def test_RCModel_sizes_2dim() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = HarrRandomReservoir(encoder, 0)
    observable = Observable(2)
    observable.add_operator(1.0, "Z 0")
    observables = [observable]
    subestimator = LinearRegression()
    model = RCModel(reservoir, observables, subestimator)
    X = np.zeros((10, 2))
    model.fit(X)
    X_test = np.zeros((30, 2))
    out = model.predict(X_test, 10)
    assert out.shape == (40, 2)
    assert out == pytest.approx(np.zeros((40, 2)))


def test_RCModel_sizes_2dim_2obs() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = HarrRandomReservoir(encoder, 0)
    observable1 = Observable(2)
    observable1.add_operator(1.0, "Z 0")
    observable2 = Observable(2)
    observable2.add_operator(1.0, "Z 1")
    observables = [observable1, observable2]
    subestimator = LinearRegression()
    model = RCModel(reservoir, observables, subestimator)
    X = np.zeros((10, 2))
    model.fit(X)
    X_test = np.zeros((30, 2))
    out = model.predict(X_test, 10)
    assert out.shape == (40, 2)
    assert out == pytest.approx(np.zeros((40, 2)))


def test_tracing_out_ancilla() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = CNOTReservoir(
        encoder, 3, 0
    )  # NB depth is zero so no dynamics take place
    observable = Observable(5)
    observable.add_operator(1.0, "Z 0")
    observable.add_operator(1.0, "Z 1")
    observables = [observable]
    subestimator = LinearRegression()
    model = RCModel(reservoir, observables, subestimator)

    prev = DensityMatrix(3)
    prev.set_Haar_random_state()
    expect, prev2 = model.next_reservoir_state_with_observables(prev, np.array([0, 0]))

    assert prev.get_matrix() == pytest.approx(prev2.get_matrix())
    assert expect[0] == pytest.approx(
        2
    )  # encoding state should still be |0> so expectation of sum of paulis should be 2


def test_can_predict_constant_seq() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = HarrRandomReservoir(encoder, 3)
    observable = Observable(2)
    observable.add_operator(1.0, "Z 0")
    observables = [observable]
    subestimator = LinearRegression()
    model = RCModel(reservoir, observables, subestimator)
    X = np.ones((10, 2))
    model.fit(X)
    X_test = np.ones((30, 2))
    out = model.predict(X_test, 10)
    assert out.shape == (40, 2)
    assert out == pytest.approx(np.ones((40, 2)))


def test_raises_NotFittedError_if_not_fitted() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = HarrRandomReservoir(encoder, 3)
    observable = Observable(2)
    observable.add_operator(1.0, "Z 0")
    observables = [observable]
    subestimator = LinearRegression()
    model = RCModel(reservoir, observables, subestimator)
    X = np.ones((10, 2))
    with pytest.raises(NotFittedError):
        model.predict(X, 10)


def test_observable_calculation() -> None:
    encoder = HEEncoder(1, 0)
    reservoir = CNOTReservoir(encoder, 1, 1)
    observable_anc = Observable(2)
    observable_anc.add_operator(
        1.0, "Z 1"
    )  # this measures the ancilla qubit (reversed numbering standard in qulcas)
    observable_enc = Observable(2)
    observable_enc.add_operator(1.0, "Z 0")
    observables = [observable_anc, observable_enc]
    model = RCModel(reservoir, observables, LinearRegression())
    prev = DensityMatrix(1)
    prev.set_computational_basis(1)
    expectations, new_prev = model.next_reservoir_state_with_observables(
        prev, np.random.uniform(-np.pi, np.pi, (1))
    )  # the input data doesn't matter since encoder has depth 0
    assert expectations.shape == (2,)
    assert expectations[0] == pytest.approx(-1)  # anc
    assert expectations[1] == pytest.approx(-1)  # enc
    assert new_prev.get_matrix() == pytest.approx(np.array([[0, 0], [0, 1.0]]))


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
    model = RCModel(reservoir, observables, subestimator)
    X = np.zeros((10, 2))
    model.fit(X)
    X = np.zeros((30, 2, 1))
    with pytest.raises(ValueError):
        model.fit(X)


def test_raises_error_incorrectly_shaped_X_predict() -> None:
    encoder = HEEncoder(2, 2)
    reservoir = HarrRandomReservoir(encoder, 0)
    observable1 = Observable(2)
    observable1.add_operator(1.0, "Z 0")
    observable2 = Observable(2)
    observable2.add_operator(1.0, "Z 1")
    observables = [observable1, observable2]
    subestimator = LinearRegression()
    model = RCModel(reservoir, observables, subestimator)
    X = np.zeros((10, 2))
    model.fit(X)
    X = np.zeros((30, 2, 1))
    with pytest.raises(ValueError):
        model.predict(X, 3)
    X = np.zeros((30, 3))
    with pytest.raises(ValueError):
        model.predict(X, 3)
