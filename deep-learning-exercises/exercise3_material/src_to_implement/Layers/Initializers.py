import numpy as np

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
        if isinstance(weights_shape,int):
            output = np.random.rand(weights_shape)
        else:
            output = np.random.rand(*weights_shape)
        return output

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        var = np.sqrt(2/(fan_out + fan_in))
        output = np.random.normal(0, var, size=weights_shape)
        return output

class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        var = np.sqrt(2/fan_in)
        output = np.random.normal(0, var, size=weights_shape)
        return output
