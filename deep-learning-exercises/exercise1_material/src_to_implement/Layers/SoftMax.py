import numpy as np

from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.forward_output = None

    def forward(self, input_tensor):
        input_tensor = input_tensor - np.amax(input_tensor)
        exp_tensor = np.exp(input_tensor)
        sum_value = np.sum(exp_tensor, axis=1)
        output_prob = exp_tensor/ sum_value[:,None]
        self.forward_output = output_prob
        #print(output_prob)
        return output_prob

    def backward(self, error_tensor):
        #print(error_tensor)
        value = np.sum(np.multiply(error_tensor, self.forward_output), axis = 1)
        output = self.forward_output* (error_tensor - value[:,None])

        return output

