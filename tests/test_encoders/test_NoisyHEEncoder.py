import numpy as np
import pytest

from qreservoir.encoders import Noisy_CHEEncoder


def test_depth_zero_NoisyCHEE_returns_state() -> None:
    encoder = Noisy_CHEEncoder(3, 0)
    input_vect = np.array([np.zeros(3)])
    state = encoder.get_encoding_state(input_vect)
    assert len(state.get_vector()) == 8


def test_depth_one_NoisyCHEE_returns_state() -> None:
    encoder = Noisy_CHEEncoder(3, 1)
    input_vect = np.array([np.zeros(3)])
    state = encoder.get_encoding_state(input_vect)
    assert len(state.get_vector()) == 8


def test_depth_one_NoisyCHEE_returns_state_multi_unitary() -> None:
    encoder = Noisy_CHEEncoder(3, 1, 5)
    input_vect = np.array([np.zeros(3)] * 5)
    state = encoder.get_encoding_state(input_vect)
    assert len(state.get_vector()) == 8


def test_depth_one_multi_qubit_per_feature_NoisyCHEE_returns_state() -> None:
    encoder = Noisy_CHEEncoder(3, 1, 5, 3)
    input_vect = np.array([np.zeros(3)] * 5)
    state = encoder.get_encoding_state(input_vect)
    assert len(state.get_vector()) == 512


def test_incorrect_feature_size_dim1_raises_error() -> None:
    encoder = Noisy_CHEEncoder(
        feature_num=3, depth=1, num_unitaries=1, qubits_per_feature=1
    )
    input_vect = np.array([np.zeros(3)] * 2)
    with pytest.raises(ValueError):
        encoder.get_encoding_state(input_vect)


def test_incorrect_input_size_dim2_raises_error() -> None:
    encoder = Noisy_CHEEncoder(
        feature_num=3, depth=1, num_unitaries=1, qubits_per_feature=1
    )
    input_vect = np.array([np.zeros(4)])
    with pytest.raises(ValueError):
        encoder.get_encoding_state(input_vect)


def test_encoder_returns_correct_size() -> None:
    encoder = Noisy_CHEEncoder(
        feature_num=2, depth=1, num_unitaries=2, qubits_per_feature=2
    )
    encoder.print_circuit()
    assert len(encoder) == 4


def test_returned_circuit_dimensions() -> None:
    encoder = Noisy_CHEEncoder(
        feature_num=3, depth=1, num_unitaries=2, qubits_per_feature=1
    )
    circuit = encoder.get_circuit(np.array([np.zeros(3)] * 2))
    assert circuit.get_qubit_count() == 3
    assert circuit.get_gate_count() == 21
    assert circuit.calculate_depth() == 11


def test_returned_circuit_dimensions_multi_enc_qubit() -> None:
    encoder = Noisy_CHEEncoder(
        feature_num=3, depth=1, num_unitaries=2, qubits_per_feature=4
    )
    circuit = encoder.get_circuit(np.array([np.zeros(3)] * 2))
    assert circuit.get_qubit_count() == 12
    assert circuit.get_gate_count() == 84
    assert circuit.calculate_depth() == 29
