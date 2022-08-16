import numpy as np
from tensor import Tensor
from graph import Graph, Node
from operations import Operation, Add
from loss import Loss, MSE
from sgd import SGD

class Model:
    def __init__(self, parameters):
	self.parameters = parameters

    def zero_grad(self):
	for parameter in self.parameters:
	    parameter.grad = None

    def step(self):
	for parameter in self.parameters:
	    if parameter.requires_grad:
	        parameter -= self.lr * parameter.grad

    def __repr__(self):
        return "Model: {}".format(self.parameters)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def train_on_batch(self, x, y):
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        self.zero_grad()

    def fit(self, x, y, epochs=10):
        for epoch in range(epochs):
            for x_batch, y_batch in zip(x, y):
                self.train_on_batch(x_batch, y_batch)

    def predict(self, x):
        return self.forward(x)

    def evaluate(self, x, y):
        return self.loss(self.predict(x), y).data

    def compile(self, optimizer=SGD(), loss=MSE()):
	self.optimizer = optimizer
	self.loss = loss

    def __repr__(self):
        return "Model: {}".format(self.parameters)
