import pytest
import numpy as np
from qulacs import DensityMatrix, QuantumState
from qulacs.state import partial_trace
from qreservoir.reservoirs import HarrRandomReservoir
from qreservoir.encoders import HEEncoder


def test_no_encoder_raises_error() -> None:
    with pytest.raises(ValueError):
        reservoir = HarrRandomReservoir(None, 3, 2)
        reservoir.get_reservoir_state(np.zeros(3), None)


def test_circuit_dimensions() -> None:
    reservoir = HarrRandomReservoir(None, 3, 2)
    circuit = reservoir.get_dynamics_circuit()
    assert circuit.get_qubit_count() == 5
    assert circuit.get_gate_count() == 1
    assert circuit.calculate_depth() == 1


def test_get_reservoir_state_with_encoder() -> None:
    encoder = HEEncoder(2, 1)
    reservoir = HarrRandomReservoir(encoder, 3)
    state = reservoir.get_reservoir_state(np.zeros(2), None)
    assert state.get_qubit_count() == 5
    assert state.get_squared_norm() == pytest.approx(1.0)


def test_connecting_incorrectly_sized_encoder_raises_error() -> None:
    encoder = HEEncoder(2, 1)
    reservoir = HarrRandomReservoir(None, 3, 3)
    with pytest.raises(ValueError):
        reservoir.connect_encoder(encoder)


def test_invalid_prev_raises_error() -> None:
    encoder = HEEncoder(2, 1)
    reservoir = HarrRandomReservoir(encoder, 3)

    input_vect = np.random.uniform(-np.pi, np.pi, 2)

    prev_matrix = DensityMatrix(4)
    prev_matrix.set_Haar_random_state()

    with pytest.raises(ValueError):
        _ = reservoir.get_reservoir_state(input_vect, prev_matrix)


def test_types_get_reservoir_state() -> None:
    encoder = HEEncoder(2, 1)
    reservoir = HarrRandomReservoir(encoder, 3)

    input_vect = np.zeros(2)

    prev_matrix = DensityMatrix(3)
    prev_matrix.set_Haar_random_state()

    prev_state = QuantumState(3)
    prev_state.set_Haar_random_state()

    assert type(reservoir.get_reservoir_state(input_vect, prev_matrix)) is DensityMatrix
    assert type(reservoir.get_reservoir_state(input_vect, prev_state)) is QuantumState
    assert type(reservoir.get_reservoir_state(input_vect, None)) is QuantumState
