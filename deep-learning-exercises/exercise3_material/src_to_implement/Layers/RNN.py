import copy
import numpy as np

from .FullyConnected import *
from .Sigmoid import *
from .TanH import *
from .Base import BaseLayer

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.trainable = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.last_hidden_state = None
        self.e_hidden_state = None

        self.FC_1 = FullyConnected(hidden_size + input_size, hidden_size)
        self.FC_2 = FullyConnected(hidden_size, output_size)

        self.sigmoid = Sigmoid()
        self.tanH = TanH()

        self._memorize = False
        self._optimizer_FC1 = None
        self._optimizer_FC2 = None

    @property
    def optimizer(self):
        return self._optimizer_FC1

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer_FC1 = copy.deepcopy(value)
        self._optimizer_FC2 = copy.deepcopy(value)

    @property
    def weights(self):
        return self.FC_1.weights

    @weights.setter
    def weights(self, weights):
        if hasattr(self, 'FC_1'):
            self.FC_1.weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_FC1

    @gradient_weights.setter
    def gradient_weights(self, value):
        self.FC_1.gradient_weights = value

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    # call initializer for FC layers
    def initialize(self, weights_initializer, bias_initializer):
        self.FC_2.initialize(weights_initializer, bias_initializer)
        self.FC_1.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = self.input_tensor.shape[0]

        # initialize hidden states (bs+1 to account for memorized last hidden state)
        self.hidden_states = np.zeros((batch_size + 1, self.hidden_size))

        if self.memorize:
            self.hidden_states[0] = self.last_hidden_state

        output = np.zeros((batch_size, self.output_size))

        self.FC_1_input_tensors = []
        self.FC_2_input_tensors = []

        self.TanH_output = np.zeros((batch_size, self.hidden_size))
        self.Sigmoid_output = np.zeros((batch_size, self.output_size))

        # iterate over batch size
        for i in range(batch_size):
            # stack hidden state and input tensor
            FC_1_input_tensor = np.expand_dims(np.hstack((self.input_tensor[i], self.hidden_states[i])), axis = 0)

            # compute new hidden state
            self.hidden_states[i + 1] = self.tanH.forward(self.FC_1.forward(FC_1_input_tensor))
            self.FC_1_input_tensors.append(self.FC_1.input_tensor)

            # coompute output (FC_2 followed by sigmoid function)
            output[i] = self.sigmoid.forward(
                self.FC_2.forward(np.expand_dims(self.hidden_states[i + 1], axis = 0))
            )[0]
            self.FC_2_input_tensors.append(self.FC_2.input_tensor)

            # save activation outputs
            self.TanH_output[i] = self.tanH.output
            self.Sigmoid_output[i] = self.sigmoid.output

        # save last hidden state (for BPTT)
        self.last_hidden_state = self.hidden_states[-1]
        return output

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        output = np.zeros((batch_size, self.input_size))

        self.gradient_weights_FC2 = 0
        self.gradient_weights_FC1 = 0

        hidden_state = np.zeros(self.hidden_size)

        # loop backward through time
        for i in reversed(range(batch_size)):
            # overwrite input tensors in FC layers -> to calc backward of FC correctly
            self.FC_2.input_tensor = self.FC_2_input_tensors[i]
            self.FC_1.input_tensor= self.FC_1_input_tensors[i]

            # overwrite output tensors in activations -> to cacl backward of activation correctly
            self.tanH.output = self.TanH_output[i]
            self.sigmoid.output = self.Sigmoid_output[i]

            # calculate backward of FC and activation layers
            error_tensor_FC2 = self.FC_2.backward(self.sigmoid.backward(np.expand_dims(error_tensor[i], axis=0)))
            error_tensor_FC1 = self.FC_1.backward(self.tanH.backward(error_tensor_FC2 + hidden_state))

            # gradient w.r.t. outpu
            output[i] = error_tensor_FC1[:, :self.input_size]
            # gradient w.r.t hidden_state (not saved at every iteration)
            hidden_state = error_tensor_FC1[:, self.input_size:]

            # accumulate gradient w.r.t weights of each FC layer
            self.gradient_weights_FC2 += self.FC_2.gradient_weights
            self.gradient_weights_FC1 += self.FC_1.gradient_weights

        

        if self.optimizer:
            self.FC_2.weights = self._optimizer_FC2.calculate_update(self.FC_2.weights, self.gradient_weights_FC2)
            self.FC_1.weights = self._optimizer_FC1.calculate_update(self.FC_1.weights, self.gradient_weights_FC1)

        self.weights_FC2 = self.FC_2.weights
        self.weights_FC1 = self.FC_1.weights

        return output

    def calculate_regularization_loss(self):
        if self.optimizer.regularizer is None:
            return 0

        return self.optimizer.regularizer.norm(np.concatenate(self.weights_FC1, self.weights_FC2))
