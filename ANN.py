import numpy as np

class ANN:

	def __init__(self):
		pass

	'''
	Compiles the Neural Network
	Takes the loss, optimiser and metrics as input
	loss:
	optimiser:
	metrics:
	'''
	def compile(self, loss, optimiser, metrics):
		pass

	'''
	This function fits/trains our model.
	It takes in the training data (X and y), the number of epochs and the batch_size.
	Note: This is where I would add in GPU progrmaming for later.
	'''
	def fit(self, X, y, epochs, batch_size):
		pass

	'''
	This function returns two things. Loss and accuracy
	First it will return the loss.  Note: I need to look at how to exactly calculate this.
	Second will be the accuracy.
	'''
	def evaluation(self, X, y)
		pass
	'''
	pass
	'''
	def backpropagate(self):
		pass

