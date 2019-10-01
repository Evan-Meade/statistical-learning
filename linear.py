'''
linear.py
Evan Meade, 2019

This script develops a linear model for a given dataset.

Linear modelling is a versatile and popular type of statistical learning
used for both regression (prediction of quantitative variables) and
classification (prediction of qualitative variables). It finds a number
of coefficients which minimize its cost function, which is often the
sum of the squares of the residuals. By treating additional terms as
separate variables, seemingly non-linear models can be optimized with
this as well. For instance, quadratic models can be found by computing
the following terms for each datapoint and assigning them to three
model terms as follows: [x^2 x 1].

Linear regression works great when the underlying distribution is thought
to be globally linear. Otherwise, one may consider another method, such as
k-nearest neighbors.

Script Function:
This script is designed to automate the tasks involved in linear modelling.
With data given in a .dat file, the script reads it in and prints an array
with the optimal linear regression coefficients.

Dataset Format:
Imported data must be stored in a .dat or .txt file where each row corresponds
to a different datapoint. There can be multiple input, output, and misc
columns in that order as shown in the example row below:
{input1} {input2} {input3} {output1} {output2} {misc1}
All values must be quantitative, and will be processed as floats.

Execution Format:
python linear.py {dataset file} {number of input cols} {number of output cols}

'''

# Built-in imports
import sys

# External library imports
import numpy as np

# Package script imports
import dat_operations as dato


'''
main()

This is the main function for the script which performs linear modelling.

First, it reads in the dataset with user-defined parameters. Then, it
calls the necessary functions to obtain the coefficient vector for the
linear model and prints it to the console.
'''
def main():
    # Read in execution arguments
    file_name = sys.argv[1]
    num_in = int(sys.argv[2])
    num_out = int(sys.argv[3])

    # Breaks down data file into x,y matrices
    indices = [num_in, num_in + num_out]
    [x_data, y_data, extra] = dato.slice_data(dato.read_dat(file_name), indices)

    # Constructs model terms from raw inputs
    x_fcn = create_x(x_data)

    # Calculates coefficients for least squares and prints
    beta = create_beta(x_fcn, y_data)
    print(beta)


'''
create_x(x_data)

Constructs model input terms to be linearly modelled from raw inputs x_data.

Iterates over each datapoint to transform inputs as needed for the desired
model to be generated.
'''
def create_x(x_data):
    # Initializes empty list
    x_fcn = []

    # Iterates over each datapoint
    for i in range(0, len(x_data)):
        # Appends inputs after converting to model input terms with fcn()
        x_fcn.append(fcn(x_data[i]))

    # Returns model terms as a matrix of floats
    x_matrix = np.matrix(x_fcn, dtype='float')
    return x_matrix


'''
fcn(x_point)

Computes model input terms as a function of input arguments at x_point.

Different fcn() lists will allow for different types of models. For a simple
linear model, use [x_point, 1]; the 1 allows for easy calculation of
the y-intercept.
'''
def fcn(x_point):
    return [x_point, 1]


'''
create_beta(x_fcn, y_data)

Finds optimal coefficients using processed inputs x_fcn and results y_data.

Essentially, this projects the actual results, y_data, onto the linear
subspace formed by the columns of x_fcn. The resulting projection can then
be used to find the optimal coefficient vector, beta. The total projection
formula used is: b = ((X`X)^-1)X`y.
'''
def create_beta(x_fcn, y_data):
    # Finds inverse of X`X since it is square
    x_inv = np.linalg.inv(np.matmul(x_fcn.T, x_fcn))

    # Multiplies remaining matrices and vectors
    xt_y = np.matmul(x_fcn.T, y_data)
    beta = np.matmul(x_inv, xt_y)

    # Returns coefficient vector beta
    return beta


# Calls main subroutine after defining all functions
if __name__ == '__main__':
    main()
