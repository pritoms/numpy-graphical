import numpy as np
from tensor import Tensor
from graph import Graph, Node
from operations import Operation, Add
from loss import Loss, MSE

class SGD:
    def __init__(self, parameters, lr=0.01):
	self.parameters = parameters
	self.lr = lr

    def zero_grad(self):
	for parameter in self.parameters:
	    parameter.grad = None

    def step(self):
	for parameter in self.parameters:
	    if parameter.requires_grad:
	        parameter -= self.lr * parameter.grad

    def __repr__(self):
        return "SGD: {}".format(self.lr)
