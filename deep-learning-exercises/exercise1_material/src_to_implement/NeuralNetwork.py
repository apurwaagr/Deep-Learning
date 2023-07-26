from Layers.Base import BaseLayer
from copy import deepcopy
import numpy as np
from Layers import *
from Optimization import Loss


class NeuralNetwork(BaseLayer):
     def __init__(self,optimizer) -> None:
          super().__init__()
          self.optimizer = optimizer
          self._data_layer = None
          self._loss_layer = None
          self.loss = []
          self.layers = []

     def append_layer(self, layer):
          if layer.trainable == True:
               layer.optimizer = deepcopy(self.optimizer)
          self.layers.append(layer)

     def forward(self):
          self.input_tensor, self.label_tensor = self.data_layer.next()
          input_tensor=self.test(self.input_tensor)
          #self.loss_layer = Loss.CrossEntropyLoss()
          loss=self.loss_layer.forward(input_tensor, self.label_tensor)
          return deepcopy(loss)

     def backward(self):
          #if (self.total_layers == 0 or not self.data_layer): return
          #input_tensor, label_tensor = self.data_layer.next()
          output = self.loss_layer.backward(self.label_tensor)
          for l in reversed(self.layers):
               output = l.backward(deepcopy(output))
          return output

     def test(self, input_tensor):
          for layer in self.layers:
               input_tensor = layer.forward(deepcopy(input_tensor))
          return input_tensor

     def train(self, iterations) -> None:
          
          for i in range(0, iterations):
             loss = self.forward()
             #print(loss)
             self.loss.append(loss)
             self.backward()
             #output=self.backward()
          return 
     
     @property
     def data_layer(self):
          return self._data_layer
     @data_layer.setter
     def data_layer(self, value):
          self._data_layer = value
          
     @property
     def loss_layer(self):
          return self._loss_layer
     @loss_layer.setter
     def loss_layer(self, value):
          self._loss_layer = value
