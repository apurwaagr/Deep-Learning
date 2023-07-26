import numpy as np
from Layers.Base import BaseLayer
from scipy import signal
import copy

class Conv(BaseLayer):
    
    def __init__(self, stride_shape: tuple, convolution_shape: list, num_kernels: int) -> None:
        super().__init__()

        # set class parameters
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        # overwrite parent parameters
        self.trainable = True

        # layer parameters 
        self.weights =  np.random.rand(self.num_kernels, *self.convolution_shape) # use * operator to unpack parameter list
        self.bias = np.random.rand(self.num_kernels)

        # private variables for gradient w.r.t. weights and bias
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        self._optimizer_weights = None
        self._optimizer_bias = None

        self._input_tensor = None
        self._conv_shape = None

        #print("shape weights is: {0}".format(self.weights.shape))


    @property
    def optimizer(self):
        return self._optimizer_weights
    

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer_weights = copy.copy(new_optimizer)
        self._optimizer_bias = copy.copy(new_optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @property
    def gradient_bias(self):
        return self._gradient_bias
    
    # 1D: b c y; 2D: b c y x
    def forward(self, input_tensor: np.array) -> np.array:

        # save input tensor for backward pass
        self._input_tensor = np.copy(input_tensor)

        # calculate the output shape (batch dimension remains, channels are # of kernels, y and x are modified based on stride)
        if not isinstance(self.stride_shape, tuple):
            if input_tensor.ndim == 3:
                output_shape = (input_tensor.shape[0], self.num_kernels, int(np.ceil(input_tensor.shape[2]/self.stride_shape[0])))
            else:
                output_shape = (input_tensor.shape[0], self.num_kernels, int(np.ceil(input_tensor.shape[2]/self.stride_shape[0])), int(np.ceil(input_tensor.shape[3]/self.stride_shape[0])))
        else:
            output_shape = (input_tensor.shape[0], self.num_kernels, int(np.ceil(input_tensor.shape[2]/self.stride_shape[0])), int(np.ceil(input_tensor.shape[3]/self.stride_shape[1])))

        # allocate memory for convolution results
        out_tensor = np.empty(output_shape)
        conv_tensor = np.empty((*output_shape[:2],*input_tensor.shape[2:]))
        self._conv_shape = conv_tensor.shape

        # perform convolution by calling correlation function
        #   loop over batches  and kernels
        for b in range(output_shape[0]):
            #   loop over kernels
            for k in range(output_shape[1]):
                # do convolution/correlation + bias
                conv_tensor[b,k] = self._correlation(input_tensor[b], self.weights[k]) + self.bias[k]

        # run stride after full convolution/correlation
        out_tensor = self._stride(conv_tensor)

        return np.copy(out_tensor)
    
    # error_tensor of shape 
    def backward(self, error_tensor:np.array) -> np.array:
        
        # do derivatives w.r.t weights
        self._weight_derivative(error_tensor)

        # do derivatives w.r.t biases
        self._bias_derivative(error_tensor)

        # do derivative w.r.t. x
        gradient_x = self._input_derivative(error_tensor)

        # call optimizer on weights
        if self._optimizer_weights:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
        
        # call optimizer of bias
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)
        
        # return gradient w.r.t input
        return np.copy(gradient_x)
    
    def initialize(self, weights_initializer, bias_initializer):
        # number of input neurons is product of channels*yk*xk:
        fan_in = np.prod(self.convolution_shape)

        # number of output neurons is product of num_kernels*yk*xk
        fan_out = np.prod(self.convolution_shape[1:])*self.num_kernels

        # call initializers for weights and bias
        self.weights = weights_initializer.initialize(self.weights.shape,fan_in=fan_in, fan_out=fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape,fan_in=fan_in, fan_out=fan_out)

    
    # performe stride  over conv_tensor of optional dimension 3-4 over dimension 3 and 4 based on class attribute stride_shape
    def _stride(self, conv_tensor: np.array) -> np.array:
        # peerform stride depending on size of stride operator and dimensions of input tensor
        if not isinstance(self.stride_shape, tuple):
            if conv_tensor.ndim == 3:
                out_tensor = conv_tensor[:,:,::self.stride_shape[0]]
            else:
                out_tensor = conv_tensor[:,:,::self.stride_shape[0],::self.stride_shape[0]]
        else:
            out_tensor = conv_tensor[:,:,::self.stride_shape[0],::self.stride_shape[1]]

        return np.copy(out_tensor)
    
    # function to insert zeros at every point in which numbers were removed previously, tensor must have same dimensions as _conv_shape
    def _unstride(self, tensor: np.array) -> np.array:

        out_tensor = np.zeros(self._conv_shape)

        if not isinstance(self.stride_shape, tuple):
            if tensor.ndim == 3:
                out_tensor[:,:,::self.stride_shape[0]] = tensor
            else:
                out_tensor[:,:,::self.stride_shape[0],::self.stride_shape[0]] = tensor
        else:
            out_tensor[:,:,::self.stride_shape[0],::self.stride_shape[1]] = tensor

        return np.copy(out_tensor)

    # input tensor of shape [c,y(,x)]
    # kernel tensor of shape [c,yk(,xk)] (kernel is smaller input tensor
    # return tensor of shape [y(,x)]
    def _correlation(self, input_tensor:np.array, kernel_tensor: np.array, padd_start: int = 1, prepend_padd_width:list = [(0,0)], padding_kernel:np.array = None, convolve:bool = False) -> np.array:
        # first zero pad x and y, then perform correlation only in valid window
        if padding_kernel is None:
            input_tensor_padded = np.pad(input_tensor, self._pad_width((np.array(kernel_tensor.shape[padd_start:])-1)/2, to_prepend=prepend_padd_width), 'constant')
        else:
            input_tensor_padded = np.pad(input_tensor, self._pad_width((np.array(padding_kernel.shape[padd_start:])-1)/2, to_prepend=prepend_padd_width), 'constant')

        #  covolve or correalte in valid window
        if convolve:
            correlated_tensor = signal.convolve(input_tensor_padded, kernel_tensor, mode = "valid")
        else:
            correlated_tensor = signal.correlate(input_tensor_padded, kernel_tensor, mode = "valid")

        return np.copy(correlated_tensor)

    def _pad_width(self, pad_size_tensor:np.array, to_prepend:list=None, to_append:list = None) -> tuple:
        
        pad_para = []
        for element in pad_size_tensor:
            #print(element)
            if element.is_integer():
                pad_para.append((int(element),int(element)))
            else:
                pad_para.append((int(np.floor(element)),int(np.ceil(element))))
                #print(pad_para)

        if to_prepend:
            for element in to_prepend:
                pad_para.insert(0,element)

        if to_append:
            for element in to_append:
                pad_para.append(element)

        return tuple(pad_para)

    # error tensor has shape [batch, num_kernels, y, x]
    # x has shape [b, c, y, x]
    # input_tensor has shape [batch, channels, y(,x)]
    # weights has shape [num_kernels,c,y,x]
    # derivative by weights is for_num_kernels(sum_over_batch(sum_over_channel_of_x(zero pad x/y by filter-1/2, do correlation x*error_tensor)))
    def _weight_derivative(self, error_tensor: np.array) -> np.array:

        # upsample error_tensor in x and y direction with zero insertion to remove effect of stride
        error_tensor_expanded = self._unstride(tensor=error_tensor)

        # extend error_tensor by extending axis 2 so that convolution has same shape -> allows to remove a for loop
        error_tensor_expanded = np.expand_dims(error_tensor_expanded, axis = (2))

        # iterate over num kernels
        for i_kernel in range(error_tensor.shape[1]):
            self._gradient_weights[i_kernel] = self._correlation(input_tensor = self._input_tensor, kernel_tensor=error_tensor_expanded[:,i_kernel], padd_start=2, prepend_padd_width= [(0,0),(0,0)], padding_kernel = self.weights)



    # derivative of bias is the sum of the error tensor over x, y and batches for num_kernels (one bias term per kernel -> one derivative per kernel)
    def _bias_derivative(self, error_tensor: np.array) -> np.array:
        if error_tensor.ndim == 3:
            self._gradient_bias = np.sum(error_tensor, axis=(0,2))
        else:
            self._gradient_bias = np.sum(error_tensor, axis=(0,2,3))

        
    # self.weights has shape [num_kernels, channels, yk(, xk)]
    # input_tensor/derivative has shape [batch, channels, y(,x)]
    # error tensor has shape [batch, num_kernels, y, x]
    #
    #   dL/dx = de/dx*dL/de
    #   for each batch  and channel: de/dx = sum_over_kernels(correlation(error, weights))
    #
    def _input_derivative(self, error_tensor: np.array) -> np.array:

        # upsample error_tensor in x and y direction with zero insertion to remove effect of stride
        error_tensor_unstrided = self._unstride(tensor=error_tensor)

        # allocate memory for input derivative
        input_derivative = np.zeros_like(self._input_tensor)

        # perform convolution for each batch and channel
        for batch in range(input_derivative.shape[0]):
            for channel in range(input_derivative.shape[1]):
                for kernel in range(error_tensor_unstrided.shape[1]):
                    input_derivative[batch, channel] += self._correlation(error_tensor_unstrided[batch, kernel], self.weights[kernel,channel], convolve=True, padd_start = 0, prepend_padd_width = None)

        return np.copy(input_derivative)
