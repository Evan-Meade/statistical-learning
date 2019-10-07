'''
simple_neural_class.py
Evan Meade, 2019

Basic classification neural network built from scratch.

Neural networks are a class of machine learning algorithms modelled loosely
off of the human brain. They are designed to detect subtle patterns through
supervised learning and a refinement of model parameters. They can be used
predictively for both classification and regression.

Execution Format:
python simple_neural_class.py

Built with reference to Konstantinos Kitsios' article on Medium:
https://towardsdatascience.com/how-to-build-a-simple-neural-network-from-scratch-with-python-9f011896d2f3
https://gitlab.com/kitsiosk/xor-neural-net

'''

# External package imports
import numpy as np


'''
sigmoid(z)

Returns normalized value of z with formula sig = 1/(1+e^(-z))
'''
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


'''
initialize_parameters(n_x, n_h, n_y)

Creates initial parameter matrices with random values.

The weight matrices are initialized with random values, but the bias matrices
are initialized as zero matrices. Each function parameter represents the
number of nodes at each layer: n_x is the input layer, n_h is the hidden
layer, and n_y is the output layer.
'''
def initialize_parameters(n_x, n_h, n_y):
    # Initializing matrices
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    # Creating dictionary of parameter matrices
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    # Returning parameter dictionary
    return parameters


'''
forward_prop(X, parameters)

Carries out algorithm to calculate output layer with current parameters.

Executes each layer with appropriate normalization and returns the normalized
layers.
'''
def forward_prop(X, parameters):
    # Reading in parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Calculates hidden layer values
    Z1 = np.dot(W1, X) + b1
    # Normalizes hidden layer values
    A1 = np.tanh(Z1)

    # Calculates output layer values
    Z2 = np.dot(W2, A1) + b2
    # Normalizes output layer values
    A2 = sigmoid(Z2)

    # Creates dictionary of normalized layers
    cache = {
        "A1": A1,
        "A2": A2
    }

    # Returns normalized layers
    return A2, cache


def calculate_cost(A2, Y):
    cost = -np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2)))/m
    cost = np.squeeze(cost)

    return cost


def backward_prop(X, Y, cache, parameters):
    A1 = cache["A1"]
    A2 = cache["A2"]

    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads


def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    new_parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return new_parameters


def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate):
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_of_iters+1):
        a2, cache = forward_prop(X, parameters)

        cost = calculate_cost(a2, Y)

        grads = backward_prop(X, Y, cache, parameters)

        parameters = update_parameters(parameters, grads, learning_rate)

        if(i%100 == 0):
            print('Cost after iteration# {:d}: {:f}'.format(i, cost))

    return parameters


def predict(X, parameters):
    a2, cache = forward_prop(X, parameters)
    yhat = a2
    yhat = np.squeeze(yhat)
    if(yhat >= 0.5):
        y_predict = 1
    else:
        y_predict = 0

    return y_predict


np.random.seed(2)

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

Y = np.array([[0, 1, 1, 0]])

m = X.shape[1]

n_x = 2
n_h = 2
n_y = 1
num_of_iters = 1000
learning_rate = 0.3

trained_parameters = model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate)

x_test = np.array([[1], [1]])
y_predict = predict(x_test, trained_parameters)

print('Neural Network prediction for example ({:d}, {:d}) is {:d}'.format(x_test[0][0], x_test[1][0], y_predict))
