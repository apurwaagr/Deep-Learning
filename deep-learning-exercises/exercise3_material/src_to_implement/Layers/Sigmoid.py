import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.output = 1.0 / (1 + np.exp(-input_tensor))
        return self.output

    def backward(self, error_tensor):
        gradient = self.output * (1 - self.output)
        return error_tensor * gradient
