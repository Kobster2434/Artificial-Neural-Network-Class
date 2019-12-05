import numpy as np
import random

class ANN:

	'''
	Class Name: ANN

	Class Description:
	This class is for a Sequential Neural Network.

	Class Parameters:
	'''
	def __init__(self):
		pass

	'''
	Function Name: compile

	Function Description:
	This function compiles the neural network (configures the learning process)

	Function parameters:
	-loss: Specifies what loss function we want to use.
	-optimiser: How we are going to optimise the weights and biases.
	'''
	def compile(self, loss, optimiser):
		pass

	'''
	Function Name: fit

	Function Description:
	This function relates to fitting/training our model.

	Function Parameters:
	-X: The training data.
	-y: The labels for the training data.
	-epochs: The number of iterations through the training data.
	-batch_size: After how many training instances do we update the training instances.
	'''
	def fit(self, X, y, epochs, batch_size):
		pass

	'''
	Function Name: evaluation

	Function Description:
	This function reuturns the loss and accuracy.

	Function Parameters:
	-X: The data without labels.
	-y: The labels to X.
	'''
	def evaluation(self, X, y)
		pass
	'''
	Function Name: backpropagate

	Function Description:
	The training process of a feed forward neural network.

	Function Parameters:
	'''
	def backpropagate(self):
		pass

	'''
	Function Name: add

	Function Description:
	This function adds a layer to the neural network.
	'''
	def add(self, t_layer):
		pass

class Dense:

	'''
	Class Name: Dense

	Class Description:
	This class is a Dense Layer (all weights present between layers).
	This class is to be passed as a parameter to ANN with the add function.

	Class Parameters:
	-units: The dimension of the output.
	-input_shape: The dimension of the input
	-activation: The activation function for this layer.
	-init_weights: How to initialise the weight(s) matrix.
	-init_bias: How to initialise the bias vector.

	note: look into regulizers.
	'''
	def __init__(self, units, activation, init_weights = "random", init_bias = "random", input_shape):
		self.units = units
		self.activation = activation
		self.input_shape = input_shape
		self.initWeights(init_weights)
		self.initBias(init_bias)

	'''
	Function Name: initWeights

	Function Description:
	Function that initalises the weights matrix.

	Function Parameter:
	-iw: This specifies how to initialise the weights.
		"random": initialised randomly betweeen zero and one.
		"random_s": initialised randomly between 0 and 0.1.
	'''
	def initWeights(iw):
		self.weights = None
		if iw == "random":
			self.weights = np.random.rand()

	'''
	Function Name: initBias

	Function Description:
	Function that initialises the bias vector.

	Function Parameter:
	-ib: This specifies how to initialise the bias vector.
		"random": initialised randomly between zero and one.
		"random_s": initialised randomly between 0 and 0.1.
	'''
	def initBias(ib):
		pass
