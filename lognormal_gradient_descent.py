'''
lognormal_gradient_descent.py
Evan Meade, 2019

'''

import sys

import numpy as np
import matplotlib.pyplot as plt


def main():
    M = 0
    S = 0
    data = np.loadtxt(sys.argv[1]).T
    x = data[0]
    y = data[1] / np.sum(data[1])


def cost(x, y, mu, sigma):
    sum = 0
    for i in range(0, len(x)):
        sum += .5 * (y[i] - yhat(x, mu, sigma)) ** 2
    return sum


def yhat(x, mu, sigma):
    pass   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


if __name__ == '__main__':
    main()
