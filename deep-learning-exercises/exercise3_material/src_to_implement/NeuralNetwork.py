from copy import deepcopy
from tqdm import tqdm
from Layers.FullyConnected  import FullyConnected
from Layers.Conv import Conv

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.loss = []
        self.layers = []
        self.optimizer = optimizer
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        # extra, required for backward pass
        self.label_tensor = None
    
    def forward(self):
        """
        Implement a method forward using input from the data layer and passing it through
        all layers of the network. Note that the data layer provides an input tensor and a
        label tensor upon calling next() on it. The output of this function should be the
        output of the last layer (i. e. the loss layer) of the network.
        """
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        
        loss = self.loss_layer.forward(input_tensor, self.label_tensor)
        
        # add regularized loss
        if not  (self.optimizer.regularizer is None):
            # find FC, CNN and RNN layers and sum weights
            for layer in self.layers:
                if isinstance(layer, FullyConnected):
                    loss += self.optimizer.regularizer.norm(layer.weights)
                if isinstance(layer, Conv):  
                    loss += self.optimizer.regularizer.norm(layer.weights)
                    loss += self.optimizer.regularizer.norm(layer.bias)
                # TODO add RNN loss

        return loss

    def backward(self):
        """
        """
        error_tensor = self.loss_layer.backward(self.label_tensor)
        # loop in reverse
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)
            # layer.update_weights(grad_weights)

    def append_layer(self, layer):
        """
        """
        if (layer.trainable):
            layer.optimizer = deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        
        self.layers.append(layer)

    def train(self, iterations):
        for iter in tqdm(range(iterations)):
            l = self.forward()
            # calculate gradients and update weights
            self.backward()
            self.loss.append(l)

        for layer in self.layers:
            # setting the phase of each of its layers
            layer.testing_phase = False
    
    def test(self, input_tensor):
        
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
            # setting the phase of each of its layers
            

        # this final input_tensor is actually the output tensor now
        return input_tensor
