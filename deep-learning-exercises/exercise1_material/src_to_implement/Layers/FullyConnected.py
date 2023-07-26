import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size) -> BaseLayer:
        # Layer is trainable
        self.trainable = True

        # allocate space and init weights
        #self.weights = np.random.rand(input_size+1, output_size)
        self.weights = np.random.rand(input_size+1, output_size)
        self.input_buffer = None
        self.error_tensor_buffer = None

        # protected members
        self._optimizer = None

    # Y = W*X+B
    def forward(self, input_tensor):
        # extend input to add bias
        input_extended = np.insert(input_tensor,input_tensor.shape[1],np.ones(input_tensor.shape[0]),axis=1)
        # calculate output and save input for backward pass
        out_tensor = np.dot(input_extended,self.weights)
        self.input_buffer = np.copy(input_extended)
        return out_tensor
    
    # dL/dX = dL/dY * dY/dX = error_tensor * W 
    def backward(self, error_tensor):
        # save data for later
        self.error_tensor_buffer = np.copy(error_tensor)
        ret_val = np.dot(error_tensor,np.transpose(self.weights))

        # optimize weights if there is an optimizer
        if self._optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        # return forward pass and remove last part of array, corresponding to bias
        return ret_val[:,:-1]
    
    # dL/dW = dY/dW * dL/dY = X * error_tensor
    @property
    def gradient_weights(self):
        return np.dot(np.transpose(self.input_buffer),self.error_tensor_buffer)

    @property
    def optimizer(self):
        return self._optimizer
    

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer