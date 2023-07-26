import numpy as np

class Sgd():

    def __init__(self, learning_rate=0.1) -> None:
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # w(i+1) = w(i) - n*grad(w(i))
        return weight_tensor - self.learning_rate*gradient_tensor