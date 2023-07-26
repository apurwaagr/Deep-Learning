import numpy as np


class Pooling:

    # constructor with arguments stride_shape and pooling_shape, with same order as conv layer
    def __init__(self, stride_shape, pooling_shape) -> None:
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False
        self.input_value = None
        self.max_loc = None

    def forward(self, input_tensor):

        self.input_value = input_tensor

        # compute shape of the output
        output_shape = (input_tensor.shape[0], input_tensor.shape[1],
                     (input_tensor.shape[2] - self.pooling_shape[0]) // self.stride_shape[0] + 1,
                     (input_tensor.shape[3] - self.pooling_shape[1]) // self.stride_shape[1] + 1)

        output = np.zeros(output_shape)
        self.max_loc = []

        for x in range(output_shape[0]):
            for y in range(output_shape[1]):
                for j in range(output_shape[2]):
                    for i in range(output_shape[3]):
                        # compute max value
                        output[x, y, j, i] = np.amax(input_tensor[x, y,
                                                     j * self.stride_shape[0]:j * self.stride_shape[0] +
                                                                              self.pooling_shape[0],
                                                     i * self.stride_shape[1]:i * self.stride_shape[1] +
                                                                              self.pooling_shape[1]])

                        # index of max value
                        max_index = np.argwhere(input_tensor[x, y,
                                                j * self.stride_shape[0]:j * self.stride_shape[0] + self.pooling_shape[
                                                    0],
                                                i * self.stride_shape[1]:i * self.stride_shape[1] + self.pooling_shape[
                                                    1]] == output[x, y, j, i]) + [j * self.stride_shape[0],
                                                                                  i * self.stride_shape[1]]

                        max_index = np.array([x, y, max_index[0][0], max_index[0][1]])

                        # list of max locations
                        self.max_loc.append(max_index)

        return output


    def backward(self, error_tensor):
        output_error = np.zeros(self.input_value.shape)
        output_shape = output_error.shape
        error_shape = error_tensor.shape
        idx = 0
        for b in range(output_shape[0]):
            for c in range(output_shape[1]):
                for j in range(0, error_shape[2]):
                    for i in range(0, error_shape[3]):
                        # add error values to the max location
                        output_error[b, c, self.max_loc[idx][2], self.max_loc[idx][3]] += error_tensor[b, c, j, i]
                        idx = idx + 1

        return output_error