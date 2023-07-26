from Layers.Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size) -> BaseLayer:
        # Layer is trainable
        self.trainable = True

        # allocate space and init weights
        self.weights = np.random.uniform(low=0, high=1,
                                         size=(input_size + 1, output_size))
        self.weights = np.random.rand(input_size + 1, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.input_buffer = None
        self.error_tensor_buffer = None

        # protected members
        self._optimizer = None

    @property
    def optimizer(self):
        """Getter of optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        """Setter of optimizer"""
        self._optimizer = opt

    @property
    def gradient_weights(self):
        """Setter of gradient weights"""
        return self._gradient_weights

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # append a column of ones for bias
        ones = np.ones((len(self.input_tensor), 1))
        self.input_tensor = np.append(self.input_tensor, ones, axis=1)
        out_tensor = np.dot(self.input_tensor, self.weights)
        self.input_buffer = np.copy(input_tensor)
        return out_tensor

    def backward(self, error_tensor):
        new_error_tensor = error_tensor @ self.weights[:-1].T  # used to pass to previous layers
        self._gradient_weights = self.input_tensor.T @ error_tensor  # used to update the current weights
        # self.gradient_bias = error_tensor
        if (self._optimizer != None):
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        return new_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        weights_shape = (self.input_size + 1, self.output_size)
        bias_shape = self.output_size
        # reinitializing the weights
        self.weights = weights_initializer.initialize(weights_shape, self.input_size, self.output_size)
        # initializing bias and storing in weight matrix
        self.weights[-1, :] = bias_initializer.initialize(bias_shape, self.input_size, self.output_size)
