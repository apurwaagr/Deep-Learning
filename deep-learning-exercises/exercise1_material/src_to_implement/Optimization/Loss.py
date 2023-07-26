import numpy as np
from scipy.special import xlogy

class CrossEntropyLoss():

    def __init__(self) -> None:
        self.prediction_buffer = None
        pass

    def forward(self, prediction_tensor, label_tensor):
        # set local variables
        loss = np.zeros(label_tensor.shape)
        epsilon = 2.22044605e-16

        # save prediction
        self.prediction_buffer = np.copy(prediction_tensor)

        # regularize prediction, so that log(0) must not be taken
        prediction_tensor_regularized = np.copy(prediction_tensor)
        prediction_tensor_regularized[prediction_tensor_regularized == 0] = epsilon

        # compute and return loss: sum(-log(yˆ)|y=1)
        loss  = np.multiply(label_tensor, np.log(prediction_tensor_regularized))
        loss= -np.sum(loss)
        return loss

    def backward(self, label_tensor):
        error_tensor = np.zeros(label_tensor.shape)
        error_tensor[label_tensor ==  1] = -1/self.prediction_buffer[label_tensor == 1]
        return error_tensor