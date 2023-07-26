from .Base import BaseLayer
import numpy as np

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()

        self.probability = probability
        self.mask = 1

    def forward(self, input_tensor):
        # no dropout in testing phase
        if (self.testing_phase):
            return input_tensor

        # create a bit mask of shape = input_tensor.shape and probability for ones self.probability
        self.mask = np.random.binomial(1, self.probability, input_tensor.shape)
        return input_tensor * self.mask / self.probability # scale out with probability to preserve activation energy

    def backward(self, error_tensor):
        #d out_tensor/d input_tensor = self.mask/self.probability
        return error_tensor * self.mask / self.probability
