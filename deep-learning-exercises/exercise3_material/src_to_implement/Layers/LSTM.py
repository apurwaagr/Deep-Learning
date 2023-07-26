import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.Sigmoid import Sigmoid
from Layers.TanH import TanH
import copy

class LSTM(BaseLayer):

    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(BaseLayer).__init__()

        self.trainable = True

        # are these needed?
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # initialize state gradients
        self.gradient_hidden_state = np.zeros(hidden_size)
        self.gradient_cell_state = np.zeros(hidden_size)

        # properties
        self._memorize = False
        self._optimizer = None
        self._gradient_weights = None

        # initialize fully connected layers
        self.forget_gate = FullyConnected(self.input_size + self.hidden_size, self.hidden_size) # sigmoid
        self.forget_activation = Sigmoid()
        self.input_gate = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)   # sigmoid
        self.input_activation = Sigmoid()
        self.update_gate = FullyConnected(self.input_size + self.hidden_size, self.hidden_size) # tanh
        self.update_activation = TanH()
        self.hidden_out_gate = FullyConnected(self.input_size + self.hidden_size, self.hidden_size) # sigmoid
        self.hidden_out_activation = Sigmoid()
        self.output_gate = FullyConnected(self.hidden_size, self.output_size) # sigmoid
        self.output_activation = Sigmoid()
        self.cell_activation = TanH()

        # initialize tensors to store internal data
        self.concatenated_tensor = None
        self.forget_tensor = None
        self.forget_tensor_fcl = None
        self.input_tensor = None
        self.input_tensor_fcl = None
        self.update_tensor = None
        self.update_tensor_fcl = None
        self.cell_state = None
        self.hidden_out_tensor = None
        self.hidden_out_tensor_fcl = None
        self.hidden_state = None
        self.out_tensor_fcl = None
        self.out_tensor = None
        self.cell_activation_tensor = None

    @property
    def gradient_weights(self):
        # get all gradient weights of shape  [input_size + hidden_size, hidden_size]
        gradient_weights_forget = self.forget_gate.gradient_weights
        gradient_weights_input = self.input_gate.gradient_weights
        gradient_weights_update = self.update_gate.gradient_weights
        gradient_weights_hidden_out = self.hidden_out_gate.gradient_weights
        return np.concatenate((gradient_weights_forget, gradient_weights_input, gradient_weights_update, gradient_weights_hidden_out),axis=1)
        
    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        size = self.hidden_size
        self.forget_gate.gradient_weights = gradient_weights[:,:size]
        self.input_gate.gradient_weights = gradient_weights[:,size:size*2]
        self.update_gate.gradient_weights = gradient_weights[:,2*size:size*3]
        self.hidden_out_gate.gradient_weights = gradient_weights[:,3*size:size*4]

    @property
    def weights(self):
        # get all weights of shape  [input_size + hidden_size, hidden_size]
        weights_forget = self.forget_gate.weights
        weights_input = self.input_gate.weights
        weights_update = self.update_gate.weights
        weights_hidden_out = self.hidden_out_gate.weights
        return np.concatenate((weights_forget, weights_input, weights_update, weights_hidden_out),axis=1)
    
    @weights.setter
    def weights(self, weights):
        size = self.hidden_size
        self.forget_gate.weights = weights[:,:size]
        self.input_gate.weights = weights[:,size:size*2]
        self.update_gate.weights = weights[:,2*size:size*3]
        self.hidden_out_gate.weights = weights[:,3*size:size*4]

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, obj):
        self.forget_gate.optimizer = copy.deepcopy(obj)
        self.input_gate.optimizer = copy.deepcopy(obj)
        self.update_gate.optimizer = copy.deepcopy(obj)
        self.hidden_out_gate.optimizer = copy.deepcopy(obj)
        self.output_gate.optimizer = copy.deepcopy(obj)

    @property
    def memorize(self):
        return self._memorize
    
    @memorize.setter
    def memorize(self, val):
        self._memorize = val

    def initialize(self, weights_initializer, bias_initializer):
        self.forget_gate.initialize(copy.deepcopy(weights_initializer),copy.deepcopy(bias_initializer))
        self.input_gate.initialize(copy.deepcopy(weights_initializer),copy.deepcopy(bias_initializer))
        self.update_gate.initialize(copy.deepcopy(weights_initializer),copy.deepcopy(bias_initializer))
        self.hidden_out_gate.initialize(copy.deepcopy(weights_initializer),copy.deepcopy(bias_initializer))
        self.output_gate.initialize(copy.deepcopy(weights_initializer),copy.deepcopy(bias_initializer))


    # input_tensor.dim = [t, input_size]
    def forward(self, input_tensor):

        DEBUG = False

        batch_size = input_tensor.shape[0]
        
        if (self.cell_state is None) or (self._memorize == False):
            self.concatenated_tensor = np.zeros((batch_size, 1, self.hidden_size +  self.input_size))
            self.forget_tensor = np.zeros((batch_size, 1, self.hidden_size))
            self.forget_tensor_fcl = np.zeros((batch_size, 1, self.hidden_size))
            self.input_tensor = np.zeros((batch_size, 1, self.hidden_size))
            self.input_tensor_fcl = np.zeros((batch_size, 1, self.hidden_size))
            self.update_tensor = np.zeros((batch_size, 1, self.hidden_size))
            self.update_tensor_fcl = np.zeros((batch_size, 1, self.hidden_size))
            self.cell_state = np.zeros((batch_size, 1, self.hidden_size))
            self.cell_activation_tensor = np.zeros((batch_size, 1, self.hidden_size))
            self.hidden_out_tensor = np.zeros((batch_size, 1, self.hidden_size))
            self.hidden_out_tensor_fcl = np.zeros((batch_size, 1, self.hidden_size))
            self.hidden_state = np.zeros((batch_size, 1, self.hidden_size))
            self.out_tensor_fcl = np.zeros((batch_size, 1, self.output_size))
            self.out_tensor = np.zeros((batch_size, self.output_size))

        for i, input_time_slice in enumerate(input_tensor):


            # input zero as hidden state at i == 0:
            if i == 0:
                self.concatenated_tensor[i] = np.expand_dims(np.concatenate((self.hidden_state[i,0], input_time_slice)),axis=0)
            else: 
                self.concatenated_tensor[i] = np.expand_dims(np.concatenate((self.hidden_state[i-1,0], input_time_slice)), axis=0)

            # foget gate
            tmp = self.forget_gate.forward(self.concatenated_tensor[i])
            self.forget_tensor_fcl[i] = self.forget_gate.forward(self.concatenated_tensor[i])
            self.forget_tensor[i] = self.forget_activation.forward(self.forget_tensor_fcl[i])
            if DEBUG:
                print("forward")
                print(self.hidden_size)
                print(self.input_size)
                print(self.output_size)
                print(self.concatenated_tensor[i].shape)
                print(self.forget_tensor_fcl[i].shape)
                print(tmp.shape)

            # input gate
            self.input_tensor_fcl[i] = self.input_gate.forward(self.concatenated_tensor[i])
            self.input_tensor[i] = self.input_activation.forward(self.input_tensor_fcl[i])
            self.update_tensor_fcl[i] = self.update_gate.forward(self.concatenated_tensor[i])
            self.update_tensor[i] = self.update_activation.forward(self.update_tensor_fcl[i])

            # updating the cell state
            if i == 0:
                self.cell_state[i] = np.multiply(self.cell_state[i], self.forget_tensor[i]) + np.multiply(self.input_tensor[i], self.update_tensor[i])
            else:
                self.cell_state[i] = np.multiply(self.cell_state[i-1], self.forget_tensor[i]) + np.multiply(self.input_tensor[i], self.update_tensor[i])
        
            # update the hidden state
            self.hidden_out_tensor_fcl[i] = self.hidden_out_gate.forward(self.concatenated_tensor[i])
            self.hidden_out_tensor[i] =  self.hidden_out_activation.forward(self.hidden_out_tensor_fcl[i])
            self.cell_activation_tensor[i] = self.cell_activation.forward(self.cell_state[i])
            self.hidden_state[i] = np.multiply(self.hidden_out_tensor[i], self.cell_activation_tensor[i])

            # compute output
            self.out_tensor_fcl[i] = self.output_gate.forward(self.hidden_state[i])
            self.out_tensor[i] = self.output_activation.forward(self.out_tensor_fcl[i])[0]

        # forward hidden state:
        if self._memorize:
            self.hidden_state[0] = self.hidden_state[-1]
            self.cell_state[0] = self.cell_state[-1]

        return np.copy(self.out_tensor)
        
    # TODO! check backward path if memorize, then forget gradient is computed worngly
    def backward(self, error_tensor):
        DEBUG = False

        # initialize grad_h = zeros, grad_c = zeros
        self.gradient_hidden_state = np.zeros((error_tensor.shape[0],1,self.hidden_size))
        self.gradient_cell_state = np.zeros((error_tensor.shape[0],1,self.hidden_size))
        self.gradient_input = np.zeros((error_tensor.shape[0],self.input_size))

        for i, error_time_slice in reversed(list(enumerate(error_tensor))):
            
            # gradient output activiation
            gradient_output_activation = self.output_activation.backward(error_time_slice)
            # gradient output FCL
            gradient_output_fcl = self.output_gate.backward(gradient_output_activation)
            if DEBUG:
                print("####################################")
                print("####################################")
                print("output gradient calculation:")
                print(error_time_slice.shape)
                print(gradient_output_activation.shape)
                print(self.out_tensor_fcl[i].shape)
                print(self.output_gate.input_tensor)
                print(self.output_gate.input_tensor.shape)
                

            # gradient hidden state
            gradient_hidden_state = gradient_output_fcl + self.gradient_hidden_state[i]
            if DEBUG:
                print("computing gradient hidden state")
                print(gradient_output_fcl.shape)
                print(self.gradient_hidden_state[i].shape)

            # gradient cell activation tensor
            gradient_cell_activation_tensor = np.multiply(gradient_hidden_state, self.hidden_out_tensor[i])
            # gradient cell activation
            gradient_cell_activation = self.cell_activation.backward(gradient_cell_activation_tensor)
            # gradient cell state
            gradient_cell_state = gradient_cell_activation + self.gradient_cell_state[i]
            if DEBUG:
                print("computing gradient cell state")
                print(gradient_hidden_state.shape)
                print(self.hidden_out_tensor[i].shape)
                print(gradient_cell_activation_tensor.shape)

            # gradient hidden out tensor
            gradient_hidden_out_tensor = np.multiply(gradient_hidden_state, self.cell_activation_tensor[i])
            # gradient hidden out activation
            gradient_hidden_out_activation = self.hidden_out_activation.backward(gradient_hidden_out_tensor)
            # gradient hidden out FCL
            gradient_hidden_out_fcl = self.hidden_out_gate.backward(gradient_hidden_out_activation)

            # gradient update gate tensor
            gradient_update_gate_tensor = np.multiply(self.input_tensor[i],gradient_cell_state)
            # gradient update gate activation
            gradient_update_gate_activation = self.update_activation.backward(gradient_update_gate_tensor)
            # gradient update gate FCL
            gradient_update_gate_fcl = self.update_gate.backward(gradient_update_gate_activation)
            if DEBUG:
                print("computing gradient update gate")
                print(self.input_tensor[i].shape)
                print(gradient_cell_state.shape)
                print(gradient_update_gate_tensor.shape)
                print(gradient_update_gate_activation.shape)
                print(gradient_update_gate_fcl.shape)

            # gradient input gate tensor
            gradient_input_gate_tensor = np.multiply(self.update_tensor[i], gradient_cell_state)
            # gradient input gate activation
            gradient_input_gate_activation = self.input_activation.backward(gradient_input_gate_tensor)
            # gradient input gate FCL
            gradient_input_gate_fcl = self.input_gate.backward(gradient_input_gate_activation)

            if i > 0:
                # gradient forget gate tensor
                gradient_forget_gate_tensor = np.multiply(self.cell_state[i-1],gradient_cell_state)
                # gradient forget gate activation
                gradient_forget_gate_activation = self.forget_activation.backward(gradient_forget_gate_tensor)
                # gradient forget gate FCL
                gradient_forget_gate_fcl = self.forget_gate.backward(gradient_forget_gate_activation)
            else:
                gradient_forget_gate_fcl = np.zeros_like(gradient_input_gate_fcl)


            gradient_x_tilde = gradient_hidden_out_fcl + gradient_update_gate_fcl + gradient_input_gate_fcl + gradient_forget_gate_fcl

            if DEBUG:
                print("backward debug")
                print(gradient_hidden_out_fcl.shape)
                print(gradient_update_gate_fcl.shape)
                print(gradient_input_gate_fcl.shape)
                print(gradient_forget_gate_fcl.shape)
                print("   adds up to")
                print(gradient_x_tilde.shape)

            # gradient concatenated x_t and h_{t-1}
            self.gradient_input[i] = gradient_x_tilde[0,self.hidden_size:]

            # gradient c_{t-1} and h_{t-1}
            if i > 0:
                self.gradient_cell_state[i-1] = np.multiply(self.forget_tensor[i], gradient_cell_state)
                self.gradient_hidden_state[i-1] =  gradient_x_tilde[0,:self.hidden_size]

        return np.copy(self.gradient_input)