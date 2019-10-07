'''
neural.py
Evan Meade, 2019



'''

import numpy as np


class model(object):
    def __init__(self, n_x, n_h, n_y):
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y

        self.gen_params()

    def gen_params(self):
        n_x = self.n_x
        n_h = self.n_h
        n_y = self.n_y

        params = []

        W0 = np.random.randn(n_h[0], n_x)
        b0 = np.zeros((n_h[0], 1))
        params.append([W0, b0])

        for i in range(1, len(n_h) - 1):
            W = np.random.randn(n_h[i], n_h[i-1])
            b = np.zeros((n_h[i], 1))
            params.append([W, b])

        Wf = np.random.randn(n_y, n_h[-1])
        bf = np.zeros((n_y, 1))
        params.append([Wf, bf])

        self.params = params

    def train(self, X, Y, iters, rate):
        pass
