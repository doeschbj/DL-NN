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
		layer5 = SimpleLayer(numberInputs=5,numberNeurons=20)
		layer6	= Sigmoid()

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
		for layer in self.layers:
			input = layer.forward(input)
			#print(input)
		return input
	def forward_oneStep(self,input):
		input = self.layers[0].forward(input)
		return input


	def backward(self, input):
		pass




if __name__=="__main__":
	model = Model()
	#Somehow normalize the input
	sampleinput = np.arange(30)#262144)
	print(model.forward(sampleinput))
	#compute_error
	#model.backward(error)
