#!/usr/bin/env python
import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        outputs =   np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))
        return outputs

    def backward(self, dout):
        sigmoid = 1 / (1 + np.exp(-dout))
        return sigmoid * (1 - sigmoid)

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        outputs = np.maximum(0,x)
        return outputs

    def backward(self, dout):
        dout[dout<0] = 0
        return dout

class LeakyReLU:
    def __init__(self, strength):
        self.strength = strength
        pass

    def forward(self, x):
        outputs = np.maximum(self.strength * x,x)
        return outputs

    def backward(self, dout):
        dx = np.ones_like(dout)
        dx[dout < 0] = self.strength
        return dx

class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        outputs = np.tanh(x)
        return outputs

    def backward(self, dout):
        tan = np.tanh(dout)
        dout = 1- tan**2
        return dout

class EmptyActFunc:
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, dout):
        return dout

class Softmax:
    def __init__(self):
        pass

    def forward(self,x):
        e = np.exp(np.subtract(x,np.max(x)))
        x = e/np.sum(e)
        return x

    def backward(self, dout):
        #tan = np.tanh(dout)
        #dout = 1- tan**2
        #return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        out = np.ones(10)
        out = out[...,None]
        return out
