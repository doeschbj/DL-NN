#!/usr/bin/env python
import numpy as np
from layer import *

class Model:
	def __init__(self):
		self.layers = []
		layer1 = SimpleLayer(numberInputs=30,numberNeurons=20) #numberNeurons euqals output of Layer numberInputs=262144
		layer2 = Sigmoid()
		layer3 = SimpleLayer(numberInputs=20,numberNeurons=5)
		layer4 = Sigmoid()
		layer5 = SimpleLayer(numberInputs=5,numberNeurons=10)
		layer6	= Softmax()

		self.layers.append(layer1)
		self.layers.append(layer2)
		self.layers.append(layer3)
		self.layers.append(layer4)
		self.layers.append(layer5)
		self.layers.append(layer6)
		print("Init")


	def forward(self,input):
		"""input is the input vector of the picture we are getting"""
		"""so if the picture is 512x512 the vector is of size 262 144"""
		"""We do an forward pass through all layers. Between each layer is an activation function"""
		for layer in self.layers:
			input = layer.forward(input)
		return input
	def forward_oneStep(self,input):
		input = self.layers[0].forward(input)
		return input


	def backward(self, dout):
		for layer in reversed(self.layers):
			dout = layer.backward(dout)
		return dout



if __name__=="__main__":
	loss_arr = []
	model = Model()
	loss = MultiClassCELoss()
	#Somehow normalize the input
	sampleinput = np.arange(30)#262144)
	sample_gt = np.zeros(10) # number of classes obviously
	sample_gt[3] = 1
	out = model.forward(sampleinput)
	loss_it = loss.forward(sample_gt, out)
	loss_arr.append(loss_it)
	print(out)
	loss_error = loss.backward(loss_it)
	x = model.backward(loss_error)
