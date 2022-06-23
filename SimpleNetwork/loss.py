#!/usr/bin/env python
import numpy as np

""" below are loss functions"""
class MeanSquaredError:
    """Class which computes the mean squared error"""
    def __init__(self):
        pass
    def forward(self, y_pred, g_truth):
        self.caache_g = g_truth
        mse = np.mean(np.square(np.subtract(g_truth, y_pred)))
        e_prime = 2*(y_pred-g_truth)/1
        return mse,e_prime
    def backward(self, y_pred):
        y = self.cache_g
        mse = 2 * np.subtract(y_pred, y)
        return mse

class CrossEntropyLoss:
    def __init__(self):
        pass
    def forward(self, y_pred, g_truth):
        self.cache = y_pred
        n_samples = g_truth.shape[0]
        logp = - np.log(y_pred[g_truth.argmax(axis=0)])
        e = np.sum(logp)/n_samples
        return e,0

    def backward(self, dout):
        return np.dot(-dout, 1/self.cache)

"""Just a helper"""
def clamp(min_val, max_val, val):
    """Clamps the value to max min"""
    return min(max_val, max(min_val, val))
