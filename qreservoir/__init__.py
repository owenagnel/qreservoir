"""

Qreserveoir is a Python package for quantum reservoir engineering and quantum extreme learning built on top of qulacs. 

Installation
=====
The easiest way to install `qreservoir` is using pip:

    $ pip install qreservoir

qreservoir requires Python version 3.10 or 3.11.

Modules
=====
There are four main subpackages in Qreservoir: `encoders`, `reservoirs`, `models` and `datasets`. Each of these 
gives access to a variety of classes to build and train reservoir and extreme learning models.

The general composition of a model is as follows:

1. An encoder, which encodes the input data into a quantum state
2. A reservoir, which evolves the quantum state to which we pass an encoder
3. A model, which is a wrapper around reservoirs allowing for prediction and training

Models take a scikit-learn estimator as an argument, which is used to train the model and make predictions. 
The model also takes a list of qulacs observables as an argument, which are used to calculate the expectation 
values and translate the quantum dynamics model to a classical value.

Example
======
    from qreservoir.models.QELModel import QELModel
    from qreservoir.reservoirs.RotationReservoir import RotationReservoir
    from qreservoir.encoders.HEEncoder import HEEncoder
    from qulacs import Observable
    from sklearn.linear_model import LinearRegression
    from qreservoir.datasets.Complex_Fourrier import Complex_Fourrier

    dataset = Complex_Fourrier()

    encoder = HEEncoder(1, 1, 1) # 1 feature, 1 layer, 1 qubit per feature
    reservoir = RotationReservoir(encoder, 2, 10)  # 2 ancilla qubits, 10 depth

    observable = Observable(3)
    observable.add_operator(1.0, "Z 2") # pauli-Z on the topmost qubit.. c.f. qulacs qubit ordering

    model = QELModel(reservoir, [observable], LinearRegression()) # observable is a qulacs Observable object
    X, _, y, _ = dataset.train_test_split()
    model.fit(X, y)
    print(model.score(X, y))

"""
