import numpy as np


class L2_Regularizer():

    def __init__(self, alpha) -> None:
        self.alpha = alpha


    #d||w||2/dw = w
    def calculate_gradient(self, weights) -> np.array:
        sub_gradient = weights*self.alpha
        return np.copy(sub_gradient)

    # ||w||2 = sum(|w|^2)
    def norm(self, weights) ->  np.array:
        norm = np.sum(np.square(np.abs(weights)))*self.alpha
        return np.copy(norm)



class L1_Regularizer():

    def __init__(self, alpha) -> None:
        self.alpha = alpha

    #d||w||1/dw = sign(w)
    def calculate_gradient(self, weights) -> np.array:  
        sub_gradient = np.sign(weights)*self.alpha
        return np.copy(sub_gradient)

    #||w||1 = sum(|w|) 
    def norm(self, weights) ->  np.array:
        norm  = np.sum(np.abs(weights))*self.alpha
        return np.copy(norm)