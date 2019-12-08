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

	note: consider updating later with the adam optimizer among others. Currenlty stochastic gradient descent by itself.
	'''
	def compile(self, loss, learning_rate):
		if loss == "mse":
			self.loss = self.mse
		self.learning_rate = learning_rate

	'''
	Function Name: fit

	Function Description:
	This function relates to fitting/training our model.

	Function Parameters:
	-X: The training data. It is expected that X will be in a numpy array. With pandas this can be achieved by using .to_numpy().
	-y: The labels for the training data (numpy array).
	-epochs: The number of iterations through the training data.
	-batch_size: After how many training instances do we update the training instances.
	'''
	def fit(self, X, y, epochs, batch_size):
		# This will call the getOutput function in each layers class and fit it this way.
		# Then we will backpropagate through everything and update the weights according
		# to the batch_size. This is repeated for each epoch.
#		while len()
			#for i in range(batch_size):
		nrow = X.shape[0]
		# note check indeces here. Do they start at 0 or 1 for dataframe in pandas?
		tc = 0 # training data counter
		bc = 0 # batch counter
		epoch_number = 1
		cumulative_error = np.zeros((self.layers[-1].units,1), dtype = int)
		while epoch_number <= epochs:
			while tc < nrow:
				while bc < batch_size and tc < nrow:
					# here get output for training instance in X and get/update the cumulative error. 
					pred = iterLayers(X[tc,:]) # check the notation inside this later. Is , needed?
					# check the below two lines are correct
					error = y - pred # y should be the same dimension
					cumulative_error = cumulative_error + error
					tc += 1
					bc += 1
				bc = 0
				# updates the weights.
				self.backpropagate(cumulative_error)
			epoch_number += 1
			tc = 0

	'''
	Function Name: iterLayers

	Function Description:
	Function that iterates through all layers in the network and returns the output for a training instance.
	This function aids the fit and euvaluation functions:

	Function Parameters:
	-inst: This is the training instance to pass thorugh the network.
	'''
	def iterLayers(self, inst):
		output = inst
		for layer in self.layers:
			output = self.layers.getOutput(output)
		return output

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
	"reverse-mode differentiation"

	Function Parameters:
	-cum_error: The cumulative error for one batch.
	'''
	def backpropagate(self, cum_error):
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
		#print("weights", self.weights)
		#print("(inputx", inputx)
		#print("bias", self.bias)
		#print(self.activation)
		#print(inputx.shape)
		return self.activation(np.dot(self.weights, inputx.reshape(inputx.shape[0], 1)) + self.bias)

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
	def sigmoid(self, x):
		return 1.0 / (1 + np.exp(-x))

	'''
	Function Name: relu

	Function Description:
	Implementation of the relu () activation function.

	Function Parameters:
	-x: if negative convert this value to 0, otherwise it is the same.

	note: has issues with "dead" relu.
	'''
	def relu(self, x):
		return max(0,x)

	'''
	Function Name: softmax

	Function Description:
	Implementation of the softmax activation function.

	Function Parameters:
	-x The value to apply this activation function to
	'''
	def softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis = 0)

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
	def elu(self, x, alpha):
		if alpha < 0:
			raise Exception("Alpha should not be negative The value of alpha was: {}".format(alpha))
		else:
			if x > 0:
				return 0
			else:
				return alpha * (np.exp(x) - 1)

def main():

	# Test that getOutput function in Dense is correct.
	d = Dense(3, 6, "sigmoid", "random", "random")
	#print(d.bias.shape)
	print("output", d.softmax(d.getOutput(np.array([0.07172362, 0.83726111, 0.11233911]))))
	#print(np.array([0.07172362, 0.83726111, 0.11233911]).reshape((3,1)))
main()
