import numpy as np
from tensor import Tensor
from graph import Graph, Node
from operations import Operation, Add
from loss import Loss, MSE
from sgd import SGD
from model import Model

class Linear(Model):
    def __init__(self, in_features, out_features):
	super().__init__([
        Tensor(np.random.randn(in_features, out_features), requires_grad=True),
        Tensor(np.random.randn(out_features), requires_grad=True)
    ])

    def forward(self, x):
        self.x = x
        return x @ self.parameters[0] + self.parameters[1]

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.grad @ self.parameters[0].T
        if self.parameters[0].requires_grad:
            self.parameters[0].grad += self.x.T @ self.grad
        if self.parameters[1].requires_grad:
            self.parameters[1].grad += np.sum(self.grad, axis=0)
        return None
