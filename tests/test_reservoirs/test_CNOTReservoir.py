import pytest
import numpy as np
from typing import cast
from qulacs import DensityMatrix, QuantumState
from qulacs.state import partial_trace
from qreservoir.reservoirs import CNOTReservoir
from qreservoir.encoders import HEEncoder


def test_no_encoder_raises_error() -> None:
    with pytest.raises(ValueError):
        reservoir = CNOTReservoir(None, 3, 1, 2)
        reservoir.get_reservoir_state(np.zeros(3), None)


def test_circuit_dimensions() -> None:
    reservoir = CNOTReservoir(None, 3, 2, 2)
    circuit = reservoir.get_dynamics_circuit()
    reservoir.print_circuit()
    assert circuit.get_qubit_count() == 5
    assert circuit.get_gate_count() == 10
    assert circuit.calculate_depth() == 10


def test_get_reservoir_state_with_encoder_zeroes() -> None:
    encoder = HEEncoder(2, 1)
    reservoir = CNOTReservoir(encoder, 3, 10)
    state = reservoir.get_reservoir_state(np.zeros(2), None)
    final_state = np.zeros(32)
    final_state[0] = 1.0
    assert cast(QuantumState, state).get_vector() == pytest.approx(final_state)


def test_connecting_incorrectly_sized_encoder_raises_error() -> None:
    encoder = HEEncoder(2, 1)
    reservoir = CNOTReservoir(None, 3, 1, 3)
    with pytest.raises(ValueError):
        reservoir.connect_encoder(encoder)


def test_zero_layer_reservoir() -> None:
    encoder = HEEncoder(2, 1)
    reservoir = CNOTReservoir(encoder, 3, 0)

    input_vect = np.random.uniform(-np.pi, np.pi, 2)
    encoding_state = encoder.get_encoding_state(input_vect)
    encoding_matrix = DensityMatrix(2)
    encoding_matrix.load(encoding_state)

    prev_matrix = DensityMatrix(3)
    prev_matrix.set_Haar_random_state()

    reservoir_state = reservoir.get_reservoir_state(input_vect, prev_matrix)

    prev_matrix_after_reservoir = partial_trace(
        reservoir_state, [0, 1]
    )  # count from bottom remember
    encode_matrix_after_reservoir = partial_trace(reservoir_state, [2, 3, 4])

    assert prev_matrix.get_matrix() == pytest.approx(
        prev_matrix_after_reservoir.get_matrix()
    )
    assert encoding_matrix.get_matrix() == pytest.approx(
        encode_matrix_after_reservoir.get_matrix()
    )


def test_invalid_prev_raises_error() -> None:
    encoder = HEEncoder(2, 1)
    reservoir = CNOTReservoir(encoder, 3, 1)

    input_vect = np.random.uniform(-np.pi, np.pi, 2)

    prev_matrix = DensityMatrix(4)
    prev_matrix.set_Haar_random_state()

    with pytest.raises(ValueError):
        _ = reservoir.get_reservoir_state(input_vect, prev_matrix)


def test_types_get_reservoir_state() -> None:
    encoder = HEEncoder(2, 1)
    reservoir = CNOTReservoir(encoder, 3, 1)

    input_vect = np.zeros(2)

    prev_matrix = DensityMatrix(3)
    prev_matrix.set_Haar_random_state()

    prev_state = QuantumState(3)
    prev_state.set_Haar_random_state()

    assert type(reservoir.get_reservoir_state(input_vect, prev_matrix)) is DensityMatrix
    assert type(reservoir.get_reservoir_state(input_vect, prev_state)) is QuantumState
    assert type(reservoir.get_reservoir_state(input_vect, None)) is QuantumState


def test_zero_ancilla_reservoir() -> None:
    encoder = HEEncoder(2, 1)
    reservoir = CNOTReservoir(encoder, 0, 0)
    circuit = reservoir.get_dynamics_circuit()
    assert circuit.get_qubit_count() == 2
    assert circuit.get_gate_count() == 0
    assert circuit.calculate_depth() == 0
    input_vect = np.random.uniform(-np.pi, np.pi, 2)
    encoding_state = encoder.get_encoding_state(input_vect)
    reservoir_state = reservoir.get_reservoir_state(input_vect, None)
    assert cast(QuantumState, reservoir_state).get_vector() == pytest.approx(
        encoding_state.get_vector()
    )


def test_full_state_evolution() -> None:
    encoder = HEEncoder(2, 1)
    reservoir = CNOTReservoir(encoder, 3, 1)
    input_vect = np.zeros(2)
    prev = DensityMatrix(3)
    prev.set_computational_basis(5)
    final_state = reservoir.get_reservoir_state(input_vect, prev)
    final_state_analytic = DensityMatrix(5)
    final_state_analytic.set_computational_basis(12)
    assert cast(DensityMatrix, final_state).get_matrix() == pytest.approx(
        final_state_analytic.get_matrix()
    )
