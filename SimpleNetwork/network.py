#!/usr/bin/env python
import numpy as np
from layer import *
from optimizer import *
import csv

class Network:
    """This class defines the NN and holds the one hidden layer."""
    def __init__(self, alpha, optimizer):
        """init function says which method is used to init the weights
        1 == random | 2 == all weights are 1/12  | 3 == small random numbers | else == all weighte are zero"""
        init_function = 3

        #learning rate
        self.alpha = alpha
        self.layers = []
        self.opt = self.getOptimizer(optimizer, alpha)

        """add layers like this, activations can be:"Sigmoid", "ReLU",
        "Tanh","LReLU","Softmax"""
        self.addFCLayer(784, 128, init_function, "Sigmoid")
        self.addFCLayer(128, 10, init_function, "Softmax")

    def forward_step(self, input):
        """computes the forward pass of all layers by iterating through them abd calling the forward function of each layer"""
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward_step(self, dout):
        #backward pass and adjust weights for all layers
        for layer in reversed(self.layers):
            dout, dw, db = layer.backward(dout)
            weights, bias = self.opt.updateWeights(dw, db, layer.getWeightMatrix(), layer.getBiasVector())
            layer.load_weights(weights, bias)

    def addFCLayer(self, number_input, number_output, init_function, activation):
        self.layers.append(FullyConnectedLayer(number_input, number_output, init_function, activation))

    def save_weights(self):
        """This function stores the trained weights into an csv file.
            After training is done the weights can be loaded from that file instead of learn again.
            We have one csv file with weights trained on all 150 data points and one file with weights
            trained on 75 data points (use_split)"""
        header=["weights"]
        name = './weights/weights.csv'

        with open(name, 'w', newline='') as outfile:
            writer=csv.writer(outfile)
            writer.writerow(header)
            for layer in self.layers:
                weight_store = layer.getWeightMatrix()
                bias = layer.getBiasVector()
                for row in weight_store:
                    writer.writerow(row)
                writer.writerow(["next"])
                for row in bias:
                    writer.writerow(row)
                writer.writerow(["next_b"])
            writer.writerow(["end"])
        print("done writing file, all weights have been written")

    def load_weights(self, weights, bias):
        """This function loads the weights into the layers"""
        i = 0
        for layer in self.layers:
            layer.load_weights(weights[i], bias[i])
            i = i+1

    def getOptimizer(self, optimizer, alpha):
        if optimizer == "GradientDescent":
            return GradientDescent(alpha)
        else:
            print("No optimizer specified. or misspelled... Default GD")
            return GradientDescent(alpha)
