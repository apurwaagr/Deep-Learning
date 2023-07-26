import numpy as np
from .Base import BaseLayer
from .Helpers import compute_bn_gradients

class BatchNormalization(BaseLayer):

    def __init__(self, channels) -> None:
        self.channels = channels

        # create gamma and beta attributes
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)

        # overwrite parent parameters
        self.trainable = True

        # public parameters
        self._testing_phase = False
        self._optimizer = None

        # internal parameters
        self.epsilon = 1e-11
        self.mean = None
        self.var = None
        self.input_buffer = None
        self.normalized_buffer = None
        self.mvg_avg_decay = 0.8
        self.gradient_bias = np.zeros_like(self.beta)
        self.gradient_weights = np.zeros_like(self.gamma)

    @property
    def bias(self):
        return self.beta
    
    @bias.setter
    def bias(self,val):
        self.beta = val

    @property
    def weights(self):
        return self.gamma
    
    @weights.setter
    def weights(self,val):
        self.gamma = val

    @property
    def optimizer(self):
        """Getter of optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        """Setter of optimizer"""
        self._optimizer = opt

    @property
    def testing_phase(self):
        """Getter of testing_phase"""
        return self._testing_phase

    @testing_phase.setter
    def testing_phase(self, bool):
        """Setter of testing_phase"""
        self._testing_phase = bool

    def initialize(self, weights_initializer, bias_initializer):
        self.gamma = weights_initializer.initialize(self.gamma.shape, self.channels, self.channels)
        self.beta = bias_initializer.initialize(self.beta.shape, self.channels, self.channels)


    # image like input_tensor.shape =  [b,c,h,w], channels = c
    # vector like input_tensor.shape = [b,x], channels = x
    def forward(self,input_tensor) -> np.array:
        
        # save input tensor:
        self.input_buffer = np.copy(input_tensor)

        # if image-like: reformat to vector_like -> process -> reformate to imge-like
        if np.ndim(input_tensor) != 2:
            tensor = self.reformat(input_tensor)
        else:
            tensor = input_tensor

        # update mean and var deviation if we are in training in a moving average fashion
        if not self.testing_phase:
            mean = np.mean(tensor, axis=0) # calc mean over batches
            var = np.var(tensor,axis=0) # calc var deviation over batches

            if self.mean is None:
                self.mean = mean
                self.var = var
            else:
                self.mean = self.mvg_avg_decay*self.mean + (1-self.mvg_avg_decay)*mean
                self.var = self.mvg_avg_decay*self.var + (1-self.mvg_avg_decay)*var
        # usse 
        else:
            mean = self.mean
            var = self.var

        # compute normalized tensor and safe result for backward function
        normalized_tensor = np.divide(tensor-mean, np.sqrt(var + self.epsilon))

        # scale normalized tensor
        scaled_normalized_tensor = self.gamma*normalized_tensor+self.beta

        # if input is image_like: reformate back to image_like
        if np.ndim(input_tensor) != 2:
            scaled_normalized_tensor = self.reformat(scaled_normalized_tensor)
            self.normalized_buffer = np.copy(self.reformat(normalized_tensor))
        else:
            self.normalized_buffer = np.copy(self.reformat(normalized_tensor))

            

        return np.copy(scaled_normalized_tensor)




    def backward(self, error_tensor : np.array) -> np.array:

        if self.input_buffer is None:
            print("ERROR: calling backward prior to forward of BatchNormalization")
            return(-1)
        
        # if image_like  -> reformat to vector_like
        if np.ndim(error_tensor) != 2:
            tensor = self.reformat(error_tensor)
            input_buffer =  self.reformat(self.input_buffer)
            normalized_buffer = self.reformat(self.normalized_buffer)
        else:
            tensor = error_tensor
            input_buffer = self.input_buffer
            normalized_buffer = self.normalized_buffer

        # compute gradient w.r.t. gamma: dL/dgamma = dy/dgamma*dL/dy=sum_over_batches(error_tensor*input)
        gradient_gamma = np.sum(np.multiply(tensor, normalized_buffer),axis=0)
        self.gradient_weights = np.copy(gradient_gamma)

        # compute gradient w.r.t beta: dL/dbeta = dL/dy*dy/dbeta = dL/dy = sum_over_batches(error_tensor)
        gradient_beta = np.sum(tensor, axis=0)
        self.gradient_bias = np.copy(gradient_beta)
        
        # call optimizer if there is one
        if not self._optimizer is None:
            
            self.gamma = self._optimizer.calculate_update(self.gamma, gradient_gamma)

            self.beta = self._optimizer.calculate_update(self.beta, gradient_beta)

        # compute gradient w.r.t input
        gradient_input = compute_bn_gradients(tensor,input_buffer, self.gamma, self.mean, self.var, eps=self.epsilon)
        
        # if image_like  -> reformat to image_like after processing
        if np.ndim(error_tensor) != 2:
            gradient_input = self.reformat(gradient_input)

        return np.copy(gradient_input)
    
    # image like input_tensor.shape =  [b,h,m,n], channels = h
    # vector like input_tensor.shape = [b,h], channels = h
    def reformat(self, input_tensor: np.array) ->  np.array:
        # vector like -> format in shape of input  format
        if np.ndim(input_tensor)==2:
            # transpose to BxM*NxH
            tensor_reformated_0 = input_tensor.reshape((-1,)+(int(np.prod(self.input_buffer.shape[2:])),)+(input_tensor.shape[-1],))
            # reformate to BxHxM*N
            tensor_reformated_1 = np.transpose(tensor_reformated_0, (0,2,1))
            # reformate to BxHxMxN
            tensor_reformated = tensor_reformated_1.reshape(tensor_reformated_1.shape[:2]+self.input_buffer.shape[2:])

        # image like -> format to tensor of [b*h*w,c]
        else:
            # reformate to BxHxM*N
            tensor_reformated_0 = input_tensor.reshape(input_tensor.shape[:2]+(-1,))
            # transpose to BxM*NxH
            tensor_reformated_1 = np.transpose(tensor_reformated_0,(0,2,1))
            # reformate to B*M*NxH
            tensor_reformated = tensor_reformated_1.reshape((-1,)+(tensor_reformated_1.shape[-1],))

        return tensor_reformated