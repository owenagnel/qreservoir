# Qreservoir

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Qreservoir is a lightweight python package built on top of qulacs to simulate quantum extreme learning and quantum reservoir computing models.

Qreservoir is licensed under the [MIT license]().

## Quick Install for Python

```
pip install qreservoir
```

Uninstall Qreservoir:

```
pip uninstall qreservoir
```

## Features

Fast simulation of quantum extreme learning machine and quantum reservoir computing. 


## Tutorial and API documents

See the following documents for tutorials.


### Python sample code

```python
from qreservoir.models.QELModel import QELModel
from qreservoir.reservoirs.HarrRandomReservoir import HarrRandomReservoir
from qreservoir.encoders.HEEncoder import HEEncoder
from qulacs import Observable
from sklearn.linear_model import LinearRegression
import numpy as np

# Define observable list
observable = Observable(2)
observable.add_operator(1.0, "Z 0")
observables = [observable]

# Define model
encoder = HEEncoder(2, 2)
reservoir = HarrRandomReservoir(encoder, 0)
subestimator = LinearRegression()
model = QELModel(reservoir, observables, subestimator)

# Training data
X = np.zeros((10, 2))
y = np.zeros(10)

# Train
model.fit(X, y)

# Predict
X_test = np.zeros((30, 2))
out = model.predict(X_test)
```


### C++ sample code

```cpp
#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>

int main(){
    QuantumState state(3);
    state.set_Haar_random_state();

    QuantumCircuit circuit(3);
    circuit.add_X_gate(0);
    auto merged_gate = gate::merge(gate::CNOT(0,1),gate::Y(1));
    circuit.add_gate(merged_gate);
    circuit.add_RX_gate(1,0.5);
    circuit.update_quantum_state(&state);

    Observable observable(3);
    observable.add_operator(2.0, "X 2 Y 1 Z 0");
    observable.add_operator(-3.0, "Z 2");
    auto value = observable.get_expectation_value(&state);
    std::cout << value << std::endl;
    return 0;
}
```

## How to cite

N/A
