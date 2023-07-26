from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        # weights with biases, initialized uniformly [0,1)
        # transposed W`
        self.weights = np.random.uniform(low=0, high=1, 
                                        size = (input_size + 1, output_size))
        # self.bias = np.random.uniform(low=0, high=1, size=output_size)
        self.fan_in = input_size
        self.fan_out = output_size
        self._optimizer = None
        self._gradient_weights = None
    
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
        """
        returns a tensor that serves as the input tensor for the next layer. 
        input tensor is a matrix with input size columns and batch size rows. 
        The batch size represents the number of inputs processed si-
        multaneously. The output size is a parameter of the layer specifying 
        the number of columns of the output.
        """
        # TO DO:
        # convert to new memory layout. Transported versions.

        # store this input_tensor for back prop
        # print("input shape before:", input_tensor.shape)
        self.input_tensor = input_tensor
        # append a column of ones for bias
        ones = np.ones((len(self.input_tensor), 1))
        self.input_tensor = np.append(self.input_tensor, ones, axis=1)
        zT = np.dot(self.input_tensor, self.weights)
        #(batch_size, out_shape)
        return zT

    
    def backward(self, error_tensor):
        """
        returns a tensor that serves as
        the error tensor for the previous layer. Quick reminder: in the backward pass we are
        going in the other direction as in the forward pass.
        Hint: if you discover that you need something here which is no longer available to you,
        think about storing it at the appropriate time.
        """

        # error_tensor (batch_size, out_size)
        # input_tensor (batch_size, input_size)
        # dW shape (input_shape+1, out_shape)
        # weights shape (input_shape+1, out_shape)
        # new error tensor shape (batch_size, input_size)

        # calculate gradients
        # we don't need the bias part in new_error_tensor
        new_error_tensor = error_tensor @ self.weights[:-1].T # used to pass to previous layers
        self._gradient_weights = self.input_tensor.T @ error_tensor # used to update the current weights
        # self.gradient_bias = error_tensor
        if (self._optimizer != None):
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            # self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
        # print("input tensor shape:", self.input_tensor.shape)
        # print("error tensor shape:", error_tensor.shape)
        # print("weights shape:", self.weights.shape)     
        # print("dW shape:", self.dW.shape)
   

        return new_error_tensor
    

    def initialize(self, weights_initializer, bias_initializer):
        weights_shape = (self.fan_in + 1, self.fan_out)
        bias_shape = self.fan_out
        self.weights = weights_initializer.initialize(weights_shape, self.fan_in, self.fan_out)
        self.weights[-1, :] = bias_initializer.initialize(bias_shape, self.fan_in, self.fan_out)
