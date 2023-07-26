import numpy as np
import matplotlib.pyplot as plt


class Checker:
    output = np.array([])

    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        zeros = np.zeros(self.tile_size, dtype='int')
        ones = np.ones(self.tile_size, dtype='int')

        # (20 / 2) / 2 = 5 (Dividing by 2 because we will be repeating 0 1 pair)
        pattern_repeat_count = int((self.resolution / self.tile_size) / 2)

        zero_row_sub_pattern = np.concatenate([zeros, ones])
        zero_row_pattern = np.tile(zero_row_sub_pattern, (1, pattern_repeat_count))
        zero_row_pattern_set = np.tile(zero_row_pattern, (self.tile_size, 1))

        one_row_sub_pattern = np.concatenate([ones, zeros])
        one_row_pattern = np.tile(one_row_sub_pattern, (1, pattern_repeat_count))
        one_row_pattern_set = np.tile(one_row_pattern, (self.tile_size, 1))

        pattern_set = np.concatenate([zero_row_pattern_set, one_row_pattern_set])

        self.output = np.tile(pattern_set, (pattern_repeat_count, 1))
        return np.array(self.output)

    def show(self):
        
        self.draw()

        plt.imshow(self.output, 'gray')
        plt.show()

class Circle:
    output = np.array([])

    # resolution (int), radius (int), position (tuple)
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        # Generate a grid
        xx, yy = np.mgrid[:self.resolution, :self.resolution]
        # Map circle on grid
        circle = (xx - self.position[1]) ** 2 + (yy - self.position[0]) ** 2
        # Convert values to true false based on condition to generate filled circle
        self.output = np.logical_not(circle > (self.radius ** 2), circle < (self.radius ** 2))
        return np.array(self.output)

    def show(self):

        self.draw()

        plt.imshow(self.output, 'gray')
        plt.show()

class Spectrum:
    output = np.array([])

    # resolution(int)
    def __init__(self, resolution):
        self.resolution = resolution

    def draw(self):
        # Creates array of evenly spaced numbers over a specified interval
        start = np.linspace([0, 0, 1], [1, 0, 0], num=self.resolution)
        end = np.linspace([0, 1, 1], [1, 1, 0], num=self.resolution)
        self.output = np.linspace(start, end, num=self.resolution)
        return np.array(self.output)

    def show(self):

        self.draw()

        plt.imshow(self.output)
        plt.show()

