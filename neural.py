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
            W = np.random.randn(nodes[i], nodes[i-1])
            b = np.zeros((nodes[i], 1))
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
        for i in range(0, iters):
            A, cache = self.forward_prop(X)
            self.backward_prop(cache, Y, rate)

    def forward_prop(self, X):
        params = self.params
        cache = []
        V = X

        for i in range(0, len(params)):
            V = self.sigmoid(params[i][0] * V + params[i][1])
            cache.append(V)

        return V, cache

    def cost(A, Y):
        pass

    def backward_prop(self, cache, Y, rate):
        params = self.params
        n = len(Y)
        delta = []

        delta.append(np.multiply(cache[-1] - Y, sigmoid_prime(cache[-1])))

        for i in range(len(cache) - 2, -1, -1):
            delta.insert(0, np.multiply(params[i][0].T * delta[0], sigmoid_prime(cache[i])))

        for i in range(0, len(delta)):
            # params[i][0] = TO-DO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            params[i][1] = params[i][1] - rate * delta[i]

        self.params = params

    def predict(self, X):
        pass
