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

# External library imports
import numpy as np


'''
read_dat(file_name, num_in, num_out)

Grabs x and y data from the specified data file and columns.
'''
def read_dat(file_name, num_in, num_out):
    # Reads in specified data file as a NumPy array
    data = np.array(np.loadtxt(file_name))

    # Splits off x and y columns as specified
    split = np.hsplit(data, [num_in, num_in + num_out])
    x_data = np.matrix(split[0], dtype='float')
    y_data = np.matrix(split[1], dtype='float')

    # Returns x and y data as a list of NumPy matrices
    return [x_data, y_data]
