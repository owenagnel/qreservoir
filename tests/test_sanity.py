"""A test file to test verify the somewhat counterintuitive numbering system in qulacs."""
import pytest
from qulacs import QuantumState, QuantumCircuit, Observable, DensityMatrix
from qulacs.state import partial_trace, tensor_product
from qulacs.gate import X, CNOT, H
import numpy as np


def test_gate_addition() -> None:
    """Test that adding gates to a circuit adds them to the end."""
    circuit = QuantumCircuit(2)
    circuit.add_gate(H(0))
    circuit.add_gate(X(1))
    state = QuantumState(2)
    state.set_computational_basis(0)
    circuit.update_quantum_state(state)
    assert state.get_vector() == pytest.approx([0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_tensor_product() -> None:
    """Test that the tensor product works as expected."""
    state1 = QuantumState(1)
    state1.set_computational_basis(0)
    state2 = QuantumState(1)
    state2.set_computational_basis(1)
    state3 = tensor_product(state1, state2)
    assert state3.get_vector() == pytest.approx([0, 1, 0, 0])


def test_observable() -> None:
    """Test that the tensor product works as expected."""
    ob0 = Observable(3)
    ob0.add_operator(1.0, "Z 0")
    ob1 = Observable(3)
    ob1.add_operator(1.0, "Z 1")
    ob2 = Observable(3)
    ob2.add_operator(1.0, "Z 2")

    state = QuantumState(3)
    state.set_computational_basis(6)
    expectations = [ob.get_expectation_value(state) for ob in [ob0, ob1, ob2]]
    assert expectations == pytest.approx([1, -1, -1])


def test_partial_trace() -> None:
    state = DensityMatrix(3)
    circuit = QuantumCircuit(3)
    circuit.add_gate(H(0))
    circuit.add_gate(CNOT(0, 1))

    circuit.update_quantum_state(state)
    traced_out = partial_trace(state, [0, 1])
    assert traced_out.get_matrix() == pytest.approx(np.array([[1.0, 0], [0, 0]]))


def test_tensor_partial() -> None:
    state1 = DensityMatrix(3)
    state1.set_Haar_random_state()
    state2 = DensityMatrix(3)
    state2.set_Haar_random_state()
    state3 = tensor_product(state1, state2)
    traced_out = partial_trace(state3, [0, 1, 2])
    assert traced_out.get_matrix() == pytest.approx(
        state1.get_matrix()
    )  # This is teh crucial bit, numbering starts at the bottom
