import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
	self.data = data
	self.requires_grad = requires_grad
	self.grad = None

    def backward(self):
	if self.grad is None:
	    self.grad = np.ones_like(self.data)

	if self.requires_grad:
	    return self.grad

    def __repr__(self):
        return "Tensor: {}".format(self.data)
