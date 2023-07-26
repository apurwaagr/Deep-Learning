import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.output = np.tanh(input_tensor)
        return self.output

    def backward(self, error_tensor):
        gradient = 1.0 - np.square(self.output)
        return error_tensor * gradient
