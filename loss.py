import numpy as np
from tensor import Tensor
from operations import Operation, Add

class Loss(Operation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        return self.forward_pass()

    def backward(self):
        return self.backward_pass()

    def forward_pass(self):
        raise NotImplementedError

    def backward_pass(self):
        raise NotImplementedError

    def __repr__(self):
        return "Loss: {}".format(self.name)
