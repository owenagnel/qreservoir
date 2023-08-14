import pytest
import numpy as np
from qulacs.state import inner_product
from qulacs import QuantumState
from qreservoir.encoders.HEEncoder import HEEncoder


def test_zero_layer_HEE() -> None:
    encoder = HEEncoder(3, 0)
    input_vect = np.zeros(3)
    state = encoder.get_encoding_state(input_vect)
    assert state.get_vector() == pytest.approx(np.array([1, 0, 0, 0, 0, 0, 0, 0]))


def test_one_layer_HEE_zero_input() -> None:
    encoder = HEEncoder(3, 1)
    input_vect = np.zeros(3)
    state = encoder.get_encoding_state(input_vect)
    assert state.get_vector() == pytest.approx(np.array([1, 0, 0, 0, 0, 0, 0, 0]))


def test_one_layer_HEE_one_input() -> None:
    encoder = HEEncoder(3, 1)
    input_vect = np.array([np.pi] * 3)
    state = encoder.get_encoding_state(input_vect)
    all_one_state = QuantumState(3)
    all_one_state.set_computational_basis(7)
    dot_product = np.abs(inner_product(state, all_one_state))
    assert dot_product == pytest.approx(1)


def test_incorrect_feature_size_raises_error() -> None:
    encoder = HEEncoder(3, 1)
    input_vect = np.zeros(4)
    with pytest.raises(ValueError):
        encoder.get_encoding_state(input_vect)


def test_encoder_returns_correct_size() -> None:
    encoder = HEEncoder(3, 1)
    assert len(encoder) == 3


def test_returned_circuit_dimensions() -> None:
    encoder = HEEncoder(3, 1)
    circuit = encoder.get_circuit(np.zeros(3))
    assert circuit.get_qubit_count() == 3
    assert circuit.get_gate_count() == 6
    assert circuit.calculate_depth() == 4
