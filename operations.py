import numpy as np
from tensor import Tensor

class Operation:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repres__(self):
        return "Operation: {}".format(self.name)

class Add(Operation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y

    def backward(self):
        if self.x.requires_grad:
            self.x.grad += self.grad

        if self.y.requires_grad:
            self.y.grad += self.grad
        return None
