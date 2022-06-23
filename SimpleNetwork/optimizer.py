#!/usr/bin/env python
import numpy as np


class GradientDescent:
    def __init__(self, alpha):
        self.alpha = alpha

    def updateWeights(self, dw, db, weights, bias):
        weights = weights - self.alpha * dw
        bias = bias - self.alpha * db
        return weights, bias
