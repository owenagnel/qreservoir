"""

Qreserveoir is a Python package for quantum reservoir engineering and quantum extreme learning built on top of qulacs. 

Installation
=====
The easiest way to install `qreservoir` is using pip:

    $ pip install qreservoir

`qreservoir` requires Python version 3.10 or 3.11.

Modules
=====
There are four main modules in `qreservoir`: `encoders`, `reservoirs`, `models` and `datasets`. Each of these 
gives access to a variety of classes to build and train reservoir and extreme learning models.

The general composition of a model is as follows:

1. An *encoder*, which encodes the input data into a quantum state
2. A *reservoir*, which evolves the quantum state to which we pass an encoder
3. A *model*, which is a wrapper around reservoirs allowing for prediction and training

Models take a scikit-learn estimator as an argument, which is used to train the model and make predictions. 
The model also takes a list of qulacs observables as an argument, which are used to calculate the expectation 
values and translate the quantum dynamics model to a classical value.

Example
======
    from qreservoir.models import QELModel
    from qreservoir.reservoirs import RotationReservoir
    from qreservoir.encoders import ExpEncoder
    from qulacs import Observable
    from sklearn.linear_model import LinearRegression
    from qreservoir.datasets import Complex_Fourrier


    dataset = Complex_Fourrier(complexity=1, size=1000, noise=0.0)

    encoder = ExpEncoder(1, 1, 3) # 1 feature, 1 layer, 1 qubit per feature
    reservoir = RotationReservoir(encoder, 0, 10)  # 0 ancilla qubits, 10 depth

    observables = [Observable(3) for _ in range(9)] # create observable set
    for i, ob in enumerate(observables[:3]):
        ob.add_operator(1.0, f"X {i}")
    for i, ob in enumerate(observables[3:6]):
        ob.add_operator(1.0, f"Z {i}")
    for i, ob in enumerate(observables[6:]):
        ob.add_operator(1.0, f"Y {i}")

    model = QELModel(reservoir, observables, LinearRegression()) # observable is a qulacs Observable object
    X, _, y, _ = dataset.get_train_test()
    model.fit(X, y)
    print(model.score(X, y))

"""
