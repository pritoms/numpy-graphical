import numpy as np
from loss import Loss, Operation, Add
from tensor import Tensor
from graph import Graph, Node

class MSE(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_pass(self):
        return np.sum((self.y_pred - self.y_true) ** 2)

    def backward_pass(self):
        return 2 * (self.y_pred - self.y_true)
