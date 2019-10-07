'''
simple_neural_class.py
Evan Meade, 2019

Basic classification neural network based on online tutorial.

Neural networks are a class of machine learning algorithms modelled loosely
off of the human brain. They are designed to detect subtle patterns through
supervised learning and a refinement of model parameters. They can be used
predictively for both classification and regression.

Each network consists of an input layer, a series of hidden layers, and an
output layer in sequence. Each layer is composed of a number of weighted
nodes whose weight is calculated based on all of the nodes in the previous
layer. Such calculation is computed by multiplying the previous layer vector
by a weight matrix and adding a bias vector. Then, results are normalized by
some function such as tanh or sigmoid. Essentially, each layer is computed
as follows:
    x[n+1] = sigmoid(x[n] * W[n] + b[n])
Then, supervised learning with a refinement algorithm known as backpropogation
is used to hone the model parameters to fit the training data. Predictions
can be made simply by feeding in a new input vector.

NOTE: The model appears to perform poorly when the random function is not
seeded to 2. The reason for this is unclear since it trains for every case
many times.

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
    # Initializes matrices
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    # Creates dictionary of parameter matrices
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    # Returns parameter dictionary
    return parameters


'''
forward_prop(X, parameters)

Carries out algorithm to calculate output layer with current parameters.

Executes each layer with appropriate normalization and returns the normalized
layers.
'''
def forward_prop(X, parameters):
    # Reads in parameters
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

'''
calculate_cost(A2, Y)

Calculates cost of a prediction using the cross entropy loss function.
'''
def calculate_cost(A2, Y):
    cost = -np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2)))/m
    cost = np.squeeze(cost)   # Eliminates axes of dimension 1

    return cost


'''
backward_prop(X, Y, cache, parameters)

Calculates gradients of model parameters using backpropogation.

Backpropogation is a method in supervised learning to refine model parameters
to better fit training cases. It is essentially minimizing the cost function
by gradient descent in high dimensions.
'''
def backward_prop(X, Y, cache, parameters):
    # Loads normalized layer values
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Loads weight parameters
    W2 = parameters["W2"]

    # Backpropogation calculations
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    # Creates gradient dictionary
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    # Returns gradients
    return grads


'''
update_parameters(parameters, grads, learning_rate)

Adjusts parameters using gradients from backpropogation algorithm.

Parameter matrices are updated by subtracting gradients with a coefficient
of a "learning rate" to regulate the speed of descent.
'''
def update_parameters(parameters, grads, learning_rate):
    # Loads in parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loads in gradients
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Updates parameters with gradients
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    # Creates dictionary of updated parameters
    new_parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    # Returns updated parameters
    return new_parameters


'''
model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate)

Creates neural network model trained on the given data and parameters.

First, the function initializes random parameters. Then, it trains the model
over all training cases for the number of iterations given. Backpropogation
occurs at the learning rate specified. Finally, it returns the parameters of
the trained model.
'''
def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate):
    # Initializes parameters randomly
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Iterates training over all X
    for i in range(0, num_of_iters+1):
        # Calculates prediction from current parameters
        a2, cache = forward_prop(X, parameters)

        # Calculates total cost with current parameters
        cost = calculate_cost(a2, Y)

        # Calculates gradients for backpropogation
        grads = backward_prop(X, Y, cache, parameters)

        # Updates parameters with gradient descent
        parameters = update_parameters(parameters, grads, learning_rate)

        # Prints training progress every 100 iterations
        if(i%100 == 0):
            print('Cost after iteration# {:d}: {:f}'.format(i, cost))

    # Returns trained parameters
    return parameters


'''
predict(X, parameters)

Calculates prediction on test case given current parameters.

Essentially just multiplies and adds parameters in the method described in
the script header.
'''
def predict(X, parameters):
    # Calculating final output activation a2=yhat
    a2, cache = forward_prop(X, parameters)
    yhat = a2
    yhat = np.squeeze(yhat)   # Eliminates axes of dimension 1

    # Converts final activation to binary state
    if(yhat >= 0.5):
        y_predict = 1
    else:
        y_predict = 0

    # Returns prediction bit
    return y_predict


# Initializes seed of NumPy random function
# NOTE: Network shows poor performance if not seeded to 2
np.random.seed(2)

# Training data definition
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])
m = X.shape[1]   # Reperesents number of training cases

# Specifies training model parameters
n_x = 2
n_h = 2
n_y = 1
num_of_iters = 1000
learning_rate = 0.3

# Creates trained neural network over given situation
trained_parameters = model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate)

# Defines test case
x_test = np.array([[1], [1]])
# Calculates prediction from test case
y_predict = predict(x_test, trained_parameters)

# Prints results of test case
print('Neural Network prediction for example ({:d}, {:d}) is {:d}'.format(x_test[0][0], x_test[1][0], y_predict))
