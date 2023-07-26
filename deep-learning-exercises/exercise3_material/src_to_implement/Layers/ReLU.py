import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):

    def __init__(self) -> BaseLayer:
       self.trainable = False
       self.subgrad = 0.2
       self.input_buffer = None

    def forward(self, input_tensor):
        out_tensor = np.copy(input_tensor)
        self.input_buffer = np.copy(input_tensor)
        # apply relu
        out_tensor[out_tensor < 0]  = 0
        return out_tensor

    def backward(self, error_tensor):
        grad = self.input_buffer
        # calculate gradient
        grad[grad > 0] = 1
        grad[grad <= 0] = 0
        # NO SUBGRADIENTS
        #grad[grad == 0] = 0

        # caclulate error_tensor_out
        #print(error_tensor.shape)
        #print(grad.shape)

        # dL/dX = dX/dY*dL/dY = grad(X)*error_tensor
        error_tensor_out = np.multiply(grad,error_tensor)

        return error_tensor_out




    