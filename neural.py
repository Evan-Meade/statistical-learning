'''
neural.py
Evan Meade, 2019



'''

import numpy as np


class model(object):
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.gen_params()

    def gen_params(self):
        nodes = self.num_nodes

        params = []

        for i in range(1, len(nodes)):
            W = np.matrix(np.random.randn(nodes[i], nodes[i-1]))
            b = np.matrix(np.zeros((nodes[i], 1)))
            params.append([W, b])

        self.params = params

    def sigmoid(self, z):
        for i in range(0, len(z)):
            z[i] = 1 / (1 + np.exp(-z[i]))

        return z

    def sigmoid_prime(self, z):
        for i in range(0, len(z)):
            z[i] = np.exp(-z[i]) / (1 + np.exp(-z[i])) ** 2

        return z

    def train(self, X, Y, iters, rate):
        n = len(X)
        for i in range(0, iters):
            delta = []
            for j in range(0, n):
                A, cache = self.forward_prop(X[j])
                delta.append(self.backward_prop(cache, Y[j]))
            delta = np.sum(delta, axis=0) / n
            self.update_parameters(cache, delta, rate)

    def forward_prop(self, X):
        params = self.params
        cache = []
        V = np.matrix(X).T

        for i in range(0, len(params)):
            #print(V)
            cache.append(V)
            #V = self.sigmoid(params[i][0] * V + params[i][1])
            V = self.sigmoid(np.dot(params[i][0], V) + params[i][1])

        return V, cache

    def cost(A, Y):
        pass

    def backward_prop(self, cache, Y):
        params = self.params
        Y = np.matrix(Y).T
        delta = []

        delta.append(np.multiply(cache[-1] - Y, self.sigmoid_prime(cache[-1])))

        for i in range(len(cache) - 2, -1, -1):
            delta.insert(0, np.multiply(params[i][0].T * delta[0], self.sigmoid_prime(cache[i])))
            pass

        return delta

    def update_parameters(self, cache, delta, rate):
        params = self.params

        for i in range(0, len(delta)):
            params[i][0] = params[i][0] - rate * np.dot(cache[i], delta[i].T)
            params[i][1] = params[i][1] - rate * delta[i]

    def predict(self, X):
        yhat, cache = self.forward_prop(X)
        print(yhat)   # Appears to be returning a 2x2 array, dimension error?
        for i in range(0, len(yhat)):
            yhat[i] = round(yhat[i])
        return yhat

test = model([2, 2, 1])
print(test.params)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
test.train(X, Y, 1000, .3)
print(test.params)
x_test = [1, 1]
print(test.predict(x_test))

# test = model([2, 2, 1])
# print(test.params)
# x = [1, 0]
# a, cache = test.forward_prop(x)
# print(a)
