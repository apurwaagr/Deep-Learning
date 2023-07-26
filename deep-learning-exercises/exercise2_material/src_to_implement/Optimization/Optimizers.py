import numpy as np

class Sgd():

    def __init__(self, learning_rate=0.1) -> None:
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # w(i+1) = w(i) - n*grad(w(i))
        return weight_tensor - self.learning_rate*gradient_tensor

class SgdWithMomentum():

    def __init__(self, learning_rate, momentum_rate) -> None:
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

        # local variables
        self.momentum_tensor = None

    def calculate_update(self, weight_tensor, gradient_tensor):

        # compute momentum tensor
        if self.momentum_tensor is None:
            self.momentum_tensor = - self.learning_rate * gradient_tensor
        else:
            self.momentum_tensor = self.momentum_rate * self.momentum_tensor - self.learning_rate * gradient_tensor

        # compute weight update
        weight_update = weight_tensor + self.momentum_tensor

        return weight_update
        

class Adam():

    def __init__(self, learning_rate, mu, rho) -> None:
        self.learning_rate = learning_rate
        # beta 1
        self.mu = mu
        # beta 2
        self.rho = rho

        # local variables
        self.momentum_tensor = None
        self.rate_tensor = None
        self.iteration = 1 # set to one, otherwise bias correction will divide by zero
        self.epsilon = 1e-8

    def calculate_update(self, weight_tensor, gradient_tensor):

        # first iteration
        if self.momentum_tensor is None:
            # calculate momentum and rate tensors
            self.momentum_tensor = (1-self.mu)*gradient_tensor
            self.rate_tensor = (1-self.rho)*np.multiply(gradient_tensor,gradient_tensor)

        # subsequent iterations
        else:
            self.momentum_tensor = self.mu*self.momentum_tensor + (1-self.mu)*gradient_tensor
            self.rate_tensor = self.rho*self.rate_tensor + (1-self.rho)*np.multiply(gradient_tensor,gradient_tensor)

        # compute bias corrections
        momentum_tensor_bias_correct = self.momentum_tensor/(1-self.mu**self.iteration)
        rate_tensor_bias_correct = self.rate_tensor/(1-self.rho**self.iteration)

        # update iteration
        self.iteration = self.iteration +1

        # compute weight update
        weight_update = weight_tensor - self.learning_rate * np.divide(momentum_tensor_bias_correct,np.sqrt(rate_tensor_bias_correct)+self.epsilon)

        return weight_update
    
