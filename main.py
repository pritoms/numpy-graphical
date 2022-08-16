import numpy as np
from tensor import Tensor
from graph import Graph, Node
from operations import Operation, Add
from loss import Loss, MSE
from sgd import SGD
from model import Model
from linear import Linear

if __name__ == "__main__":
    x = Tensor(np.array([[1, 1], [0, 1], [1, 0], [0, 0]]))
    y = Tensor(np.array([[1, 0], [1, 0], [1, 0], [0, 1]]))
    model = Linear(2, 2)
    model.compile(optimizer=SGD([model.parameters[0], model.parameters[1]], lr=0.01), loss=MSE())
    model.fit(x, y)
    print(np.round(model.predict(x).data, 3))
