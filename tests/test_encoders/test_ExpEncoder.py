import numpy as np
import pytest
from qulacs import QuantumState
from qulacs.state import inner_product

from qreservoir.encoders import ExpEncoder


def test_zero_layer_Exp() -> None:
    encoder = ExpEncoder(3, 0)
    input_vect = np.zeros(3)
    state = encoder.get_encoding_state(input_vect)
    assert state.get_vector() == pytest.approx(np.array([1, 0, 0, 0, 0, 0, 0, 0]))


def test_one_layer_Exp_zero_input() -> None:
    encoder = ExpEncoder(3, 1)
    input_vect = np.zeros(3)
    state = encoder.get_encoding_state(input_vect)
    assert state.get_vector() == pytest.approx(np.array([1, 0, 0, 0, 0, 0, 0, 0]))


def test_one_layer_HEE_zero_input_multi_enc_qubit() -> None:
    encoder = ExpEncoder(2, 1, 2)
    input_vect = np.zeros(2)
    state = encoder.get_encoding_state(input_vect)
    assert state.get_vector() == pytest.approx(
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    )


def test_one_layer_HEE_one_input() -> None:
    encoder = ExpEncoder(3, 1)
    input_vect = np.array([np.pi] * 3)
    state = encoder.get_encoding_state(input_vect)
    all_one_state = QuantumState(3)
    all_one_state.set_computational_basis(7)
    dot_product = np.abs(inner_product(state, all_one_state))
    print(state.get_vector())
    assert dot_product == pytest.approx(1)


def test_one_layer_HEE_one_input_multi_enc_qubit() -> None:
    encoder = ExpEncoder(2, 1, 2)
    input_vect = np.array([np.pi] * 2)
    state = encoder.get_encoding_state(input_vect)
    all_one_state = QuantumState(4)
    all_one_state.set_computational_basis(15)
    dot_product = np.abs(inner_product(state, all_one_state))
    assert dot_product == pytest.approx(1)


def test_one_layer_HEE_one_input_multi_enc_qubit_diff_in() -> None:
    encoder = ExpEncoder(2, 1, 2)
    input_vect = np.array([np.pi, 0.0])
    state = encoder.get_encoding_state(input_vect)
    all_one_state = QuantumState(4)
    all_one_state.set_computational_basis(3)
    print(state.get_vector())
    dot_product = np.abs(inner_product(state, all_one_state))
    assert dot_product == pytest.approx(1)


def test_incorrect_feature_size_raises_error() -> None:
    encoder = ExpEncoder(3, 1)
    input_vect = np.zeros(4)
    with pytest.raises(ValueError):
        encoder.get_encoding_state(input_vect)


def test_encoder_returns_correct_size() -> None:
    encoder = ExpEncoder(3, 1)
    assert len(encoder) == 3


def test_returned_circuit_dimensions() -> None:
    encoder = ExpEncoder(3, 1)
    circuit = encoder.get_circuit(np.zeros(3))
    assert circuit.get_qubit_count() == 3
    assert circuit.get_gate_count() == 6
    assert circuit.calculate_depth() == 4


def test_returned_circuit_dimensions_multi_enc_qubit() -> None:
    encoder = ExpEncoder(3, 1, 4)
    circuit = encoder.get_circuit(np.zeros(3))
    assert circuit.get_qubit_count() == 12
    assert circuit.get_gate_count() == 24
    assert circuit.calculate_depth() == 13


# test that the circuit returns the same state when called twice on same input
def test_circuit_reproducibility() -> None:
    encoder = ExpEncoder(3, 4)
    input_vect = np.random.uniform(0, 2 * np.pi, 3)
    state = encoder.get_encoding_state(input_vect)
    state2 = encoder.get_encoding_state(input_vect)
    assert state.get_vector() == pytest.approx(state2.get_vector())
