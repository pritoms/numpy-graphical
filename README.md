# numpy-graphical-library

This repository contains a small machine learning library that is built with NumPy. The library is currently able to compute the forward and backward pass for a linear model with a mean squared error loss function.
```


- `requirements.txt`

```markdown
numpy==1.18.1
```


## Main components
* Graph
* Node
* Tensor
* Operation
* Loss
* SGD Optimizer
* Model
* Linear Model

## Main features
* Forward and backward pass
* Stochastic gradient descent optimizer
* Mean squared error loss function

## Installation
```bash
git clone https://github.com/pritoms/numpy-graphical.git
cd numpy-graphical-library
pip install -r requirements.txt
```

## Usage
```python
from models.linear import Linear
from optimizers.sgd import SGD
from losses.mse import MSE

model = Linear(2, 2)
model.compile(optimizer=SGD([model.parameters[0], model.parameters[1]], lr=0.01), loss=MSE())
model.fit(x, y, epochs=10)
predictions = model.predict(x)
