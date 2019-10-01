'''
dat_operations.py
Evan Meade, 2019

This script contains functions to handle inputs/outputs on .dat files.

Dataset Format:
Imported data must be stored in a .dat or .txt file where each row corresponds
to a different datapoint. There can be multiple input, output, and misc
columns in that order as shown in the example row below:
{input1} {input2} {input3} {output1} {output2} {misc1}
All values must be quantitative, and will be processed as floats.

'''

# Built-in imports
import sys
import random

# External library imports
import numpy as np


'''
read_dat(file_name)

Grabs information from specified .dat file and returns it as a NumPy array.
'''
def read_dat(file_name):
    # Reads in specified data file as a NumPy array and returns it
    data = np.array(np.loadtxt(file_name))
    return data


'''
slice_data(data, indices)

Slices NumPy array into multiple matrices with cuts at given indices.

For example, to split a line [x1, x2, x3, y1, z1, z2] into x, y, z, one would
call slice_data(data, 3, 4). The function would return [x, y, z] with each
slice being represented by a matrix.
'''
def slice_data(data, indices):
    # Initializes empty list of sliced matrices
    slices = []

    # Splits off slices as specified
    split = np.hsplit(data, indices)

    # Transforms slices into matrices
    for i in range(0, len(split)):
        slices.append(np.matrix(split[i], dtype='float'))

    # Returns slices as a list of NumPy matrices
    return slices


'''
split_data(x_data, y_data, ratio)

Randomly divides given x and y data into training and testing sets.

Returns a list of [x_training, y_training, x_test, y_test] which can then be
broken up and used to evaluate a particular model.
'''
def split_data(x_data, y_data, ratio):
    # Computes parameters for sorting
    n = len(x_data)
    num_train = int(n * ratio)

    # Randomizes x and y data in same order
    data = list(zip(x_data, y_data))
    random.shuffle(data)
    [x_data, y_data] = zip(*data)

    # Returns x and y data split into training and testing sets
    return [x_data[0:num_train], y_data[0:num_train], x_data[num_train:n], y_data[num_train:n]]
