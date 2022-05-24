import numpy as np

# class Affine:
#     def __init__(self):
#         pass
#
#     def forward(self, x):
#         """
#         :param x: Inputs, of any shape
#
#         :return out: Output, of the same shape as x
#         :return cache: Cache, for backward computation, of the same shape as x
#         """
#         outputs = 1 / (1 + np.exp(-x))
#         cache = x
#         return outputs, cache
#
#     def backward(self, dout, cache):
#         """
#         :return: dx: the gradient w.r.t. input X, of the same shape as X
#         """
#         dx = None
#         dx = dout * cache * (1 - cache)
#         return dx


class SimpleLayer:
    def __init__(self, numberInputs, numberNeurons):
        self.weights = self.init_weights(numberInputs,numberNeurons) # 30x 20

    def forward(self,x):
        output = np.dot(self.weights,x)  #(outputxinput) * (input,0) = (output,0)
        return output

    def backward(self, dout):
        pass

    def init_weights(self,numberInputs, numberNeurons):
        return np.ones((numberNeurons,numberInputs)) # np.random.random((numberNeurons, numberInputs))

class MultiClassCELoss:
    def __init__(self):
        pass
    def forward(self, y, y_pred):
        #-p(input) * log(p(predicted))
        return -np.sum(np.multiply(y, np.log(y_pred)))
    def backward(self, dout):
        pass


class Softmax():
    def __init__(self):
        pass
    def forward(self,x):
        e = np.exp(x)
        x = e/np.sum(e)
        return x

    def backward(self,dout):
        pass

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        outputs = 1 / (1 + np.exp(-x))
        return outputs

    def backward(self, dout):
        pass
