import numpy as np

class BaseLayer():

    # Base class constructor, layer is by default not trainable
    def __init__(self) ->  None:
        self.trainable = False

