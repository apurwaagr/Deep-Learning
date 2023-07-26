import numpy as np

class Flatten:
    def __init__(self):
        self.input_shape = None
        self.trainable = False

    def  forward(self, input_tensor):
        # save input shape for backward pass
        self.input_shape = input_tensor.shape
        
        #return np.reshape(input_tensor,-1)
        new_input = np.ravel(input_tensor).reshape(self.input_shape[0], -1)
        return new_input

    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.input_shape)
        
