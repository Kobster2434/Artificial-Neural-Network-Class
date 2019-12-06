import numpy as np

class ANN:

	'''
	Class Name: ANN

	Class Description:
	This class is for a Sequential Neural Network.

	Class Parameters:
	'''
	def __init__(self):
		self.loss = "mse"
		self.learning_rate = 0.01
		# this will contain all of the layers in this list
		self.layers = []

	'''
	Function Name: compile

	Function Description:
	This function compiles the neural network (configures the learning process)

	Function parameters:
	-loss: Specifies what loss function we want to use.
	-learning_rate: A value specifying how fast we learn.

	note: consider updating later with the adam optimizer among others.
	'''
	def compile(self, loss, learning_rate):
		if loss == "mse":
			self.loss = self.mse
		# This is to be changes later. Default is just a placeholder.
		if optimiser == "default":
			self.optimiser = "default"
			self.learning_rate = 0.01

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
		# This will call the getOutput function in each layers class and fit it this way.
		# Then we will backpropagate through everything and update the weights according
		# to the batch_size. This is repeated for each epoch.
		pass

	'''
	Function Name: evaluation

	Function Description:
	This function reuturns the loss and accuracy.

	Function Parameters:
	-X: The data without labels.
	-y: The labels to X.
	'''
	def evaluation(self, X, y):
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
		self.layers.append(t_layer)

	'''
	Function Name: saveWeights

	Function Description:
	Function that saves the weights and bias values of the neural network.

	Function Parameters:
	-filename: The name of the file to save the weights (biases to)
	'''
	def saveWeights(self, filename):
		pass

	'''
	Function Name: readWeights

	Function Description:
	Function that reads in weights and biases from a nerual network with the format consistent to
	what saveWeights outputs.
	All layers have to be added in first to match the data to be read in.

	Function Parameters:
	-filename: The filename to read the weights from.
	'''
	def readWeights(self, filename):
		pass

	'''
	Function Name: mse

	Function Description:
	Calculates the squared error for an observation.

	Come back to this later. Do I want to average in this function or do it afterwards?
	'''
	def mse(self, y, y_hat):
		return (y - y_hat) ** 2

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
		-"sigmoid".
		-"relu".
		-"elu".
	-init_weights: How to initialise the weight(s) matrix.
	-init_bias: How to initialise the bias vector.

	note: look into regulizers.
	'''
	def __init__(self, input_shape, units, activation, init_weights = "random", init_bias = "random"):
		self.units = units
		self.activation = None
		self.input_shape = input_shape
		self.initWeights(init_weights)
		self.initBias(init_bias)
		self.setActivation(activation)

	'''
	Function Name: setActivation

	Function Description:
	Function that sets self.activation to the specified activation function given by the user.

	Function Parameter:
	-activation: string that specifies what activation function we want.
	'''
	def setActivation(self, activation):
		if activation == "sigmoid":
			self.activation = self.sigmoid
		elif activation == "relu":
			self.activation = self.relu
		elif activation == "elu":
			self.activation = self.elu

	'''
	Function Name: getOutput

	Function Description:
	Gets the output for the input to the next layer.

	Function Parameters:
	-inputx: The initial input from the data or from the previous layer.
	'''
	def getOutput(self, inputx):
		print("weights", self.weights)
		print("inputx", inputx)
		print("bias", self.bias)
		return np.dot(self.weights, inputx) + self.bias

	'''
	Function Name: initWeights

	Function Description:
	Function that initalises the weights matrix.

	Function Parameter:
	-iw: This specifies how to initialise the weights.
		"random": initialised randomly betweeen zero and one.
		"random_s": initialised randomly between 0 and 0.1.
	'''
	def initWeights(self, iw):
		self.weights = None
		if iw == "random":
			self.weights = np.random.rand(self.units, self.input_shape)
		elif iw == "random_s":
			self.weights = np.random.rand(self.units, self.input_shape) * 0.1

	'''
	Function Name: initBias

	Function Description:
	Function that initialises the bias vector.

	Function Parameter:
	-ib: This specifies how to initialise the bias vector.
		"random": initialised randomly between zero and one.
		"random_s": initialised randomly between 0 and 0.1.
	'''
	def initBias(self, ib):
		self.bias = None
		if ib == "random":
			self.bias = np.random.rand(self.units, 1)
		elif ib == "random_s":
			self.bias = np.random.rand(self.units, 1) * 0.1

	'''
	Function Name: sigmoid

	Function Description:
	Implementation of the sigmoid activation function.

	Function Paramters:
	-x: value to "squish" between zero and one.
	'''
	def sigmoid(x):
		return 1.0 / (1 + np.exp(-x))

	'''
	Function Name: relu

	Function Description:
	Implementation of the relu () activation function.

	Function Parameters:
	-x: if negative convert this value to 0, otherwise it is the same.

	note: has issues with "dead" relu.
	'''
	def relu(x):
		return max(0,x)

	'''
	Function Name: elu

	Function Description:
	Similar to dead relu but designed to eliminate the dead connection issue.
	elu (exponential linear unit) is different to relu only when dealing with negative values.
	The value is now equal to alpha * (e^x - 1).

	Function Parameters:
	-x: value that is to be "squished".
	-alpha: Parameter that can be tuned.
	'''
	def elu(x, alpha):
		if alpha < 0:
			raise Exception("Alpha should not be negative The value of alpha was: {}".format(alpha))
		else:
			if x > 0:
				return 0
			else:
				return alpha * (np.exp(x) - 1)

def main():
	d = Dense(3, 6, "test", "random_s", "random_s")
	print(d.bias.shape)
	print(d.getOutput(np.random.rand(3, 1)))

main()
