import numpy as np

#non-convex optimization problems

#Simple iniitialization scheme with a given value
class Constant:
    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        output = np.full(weights_shape, self.value)
        return output

class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        output = np.random.uniform(low=0, high=1, size=weights_shape)
        return output

#Xavier/Glorot: initialization scheme for weights
class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_out + fan_in))
        output = np.random.normal(0, sigma, size=weights_shape) #Zero-mean Gaussian
        return output


#He Initialization: Standard deviation of weights determined by size of previous layer only
class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/fan_in) #Standard Deviation
        output = np.random.normal(0, sigma, size=weights_shape) #Weights initialzed by zero-mean Gaussian
        return output
