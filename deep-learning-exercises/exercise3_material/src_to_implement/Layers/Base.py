import numpy as np

class BaseLayer():

    # Base class constructer, layer is by default not trainable
    def __init__(self) ->  None:
        self.trainable = False
        # testing_phase set to false by default
        self.testing_phase = False
