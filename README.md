# Qreservoir

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Qreservoir is a lightweight python package built on top of qulacs to simulate quantum extreme learning and quantum reservoir computing models.

Qreservoir is licensed under the [MIT license](https://github.com/owenagnel/qreservoir/blob/main/LICENSE).

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

## How to cite

N/A
