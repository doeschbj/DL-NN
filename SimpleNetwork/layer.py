#!/usr/bin/env python
import numpy as np
from activation_functions import *

class FullyConnectedLayer:
    def __init__(self, numberInputs, numberNeurons, init_funtion, act_function):
        self.weights, self.bias = self.init_weights(numberInputs,numberNeurons, init_funtion) # 30x 20
        self.bias = self.bias[...,None]

        #Init the activation function for this layer
        if act_function == "Sigmoid":
            self.act_function = Sigmoid()
        elif act_function == "ReLU":
            self.act_function = ReLU()
        elif act_function == "Tanh":
            self.act_function = Tanh()
        elif act_function == "LReLU":
            self.act_function = LeakyReLU()
        elif act_function == "Softmax":
            self.act_function = Softmax()

        else:
            #default is Sigmoid
            print("Error no activation function type specified: Set default to Sigmoid.")
            self.act_function = Sigmoid()

    def forward(self, x):
        """Computes act(input*x + bias) """
        self.cache = x
        output = np.dot(self.weights, x) + self.bias
        output = self.act_function.forward(output)
        self.out = output
        return  output

    def backward(self,  dout):
        """Computes the backward pass for this layer, and passes it to the next."""
        da = self.act_function.backward(self.out)
        dout = dout * da

        dw = np.dot(dout, np.transpose(self.cache))
        db = dout

        dout = np.dot(np.transpose(self.weights), dout)

        #self.weights = self.weights - self.alpha * dw
        #self.bias = self.bias - self.alpha * db
        return dout, dw, db

    def init_weights(self,num_input, num_output, num_init):
        """Inits weights
            1 == random
            2 == all weights are 0.25
            3 == all weights are zero """
        if num_init == 1:
            return np.random.random((num_output,num_input)), np.random.random(num_output)
        elif num_init == 2:
            return np.ones((num_output,num_input)), np.ones(num_output)
        elif num_init == 3:
            return np.random.random((num_output,num_input))/10.0, np.random.random(num_output)/10.0
        else:
            return np.zeros((num_output,num_input)), np.zeros(num_output)

    def load_weights(self, weights, bias):
        """This function somewhat loads the trained weights and overwrites its variable """
        self.weights = np.array(weights)
        self.bias = np.array(bias)

    def getWeightMatrix(self):
        return self.weights
    def getBiasVector(self):
        return self.bias
