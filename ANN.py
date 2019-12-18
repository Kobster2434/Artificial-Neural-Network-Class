import numpy as np

class ANN:

	'''
	Class Name: ANN

	Class Description:
	This class is for a Sequential Neural Network.

	Class Parameters:
	'''
	def __init__(self):
		self.loss = None
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
		nrow = X.shape[0]
		tc = 0 # training data counter
		bc = 0 # batch counter
		epoch_number = 1
		while epoch_number <= epochs:
			while tc < nrow:
				while bc < batch_size and tc < nrow:
					output = self.iterLayers(X[tc,:])
					b, w = self.backpropagate(y[tc], X[tc,:]) # maybe change later depending on what format I force for y. 
					for i in range(len(self.layers)):
						self.layers[i].weightsbp = self.layers[i].weightsbp + w[i] # error here check. added to weights not weights bp
						self.layers[i].biasbp = self.layers[i].biasbp + b[i]   # error here check
					tc += 1
					bc += 1
				bc = 0
				for layer in self.layers:
					layer.updateWB(self.learning_rate) # updates both the weights and biases.
			epoch_number += 1
			tc = 0

	'''
	Function Name: iterLayers

	Function Description:
	Function that iterates through all layers in the network and returns the output for a training instance.
	This function aids the fit and evaluation functions:

	Function Parameters:
	-inst: This is the training instance to pass thorugh the network.
	'''
	def iterLayers(self, inst):
		output = inst
		for layer in self.layers:
			output = layer.getOutput(output)
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
		#print("XSHAPE: ", X.shape)
		#print(X[1,:])
		correct = 0
		total_obs = X.shape[0]
		for i in range(total_obs): # change to 0
			output = self.iterLayers(X[i,:])
			#print("MaxTest: ", self.getMax(output))
			#print("Output:", output, "\n", "actual", y[i])
			if self.getMax(output) == y[i]:
				correct += 1
		return (0, correct / total_obs)

	def getMax(self, output):
		c_max = None
		pos = None
		for i in range(output.shape[0]):
			#print("sdfsd", output[i])
			if i == 0:
				c_max = output[0]
				pos = 0
			else:
				if output[i] > c_max:
					c_max = output[i]
					pos = i
		return pos
	'''
	Function Name: backpropagate

	Function Description:
	The training process of a feed forward neural network.
	"reverse-mode differentiation"
	'''
	def backpropagate(self, y, input_):
		weights = [np.zeros(d.weights.shape) for d in self.layers]
		bias = [np.zeros(d.bias.shape) for d in self.layers]
		outlay = self.layers[-1]
		y_vec = None
		delta = None
		#print("outlay units", outlay.units)
		if outlay.units == 1:
			y_vec = np.array([y]).reshape((1,1))
		else:
			y_vec = np.zeros((outlay.units, 1), dtype = int)
			y_vec[y][0] = 1
		delta = self.loss(y_vec, outlay.output, derivative = True) * outlay.activation(outlay.z, derivative = True)
		#print("y_hat", y_vec)
		bias[-1] = delta
		weights[-1] = np.dot(delta, self.layers[-2].output.transpose())

		# NOTE: add case where there is only one layer and below for loop isn't needed.

		for i in range(2, len(self.layers)+1):
			if i != (len(self.layers)+1):
				currlay = self.layers[-i]
				delta = np.dot(self.layers[-i+1].weights.transpose(), delta) * currlay.activation(currlay.z, derivative = True)
				bias[-i] = delta
				weights[-i] = np.dot(delta, self.layers[-i-1].output.transpose())
			else:
				currlay = self.layers[-i]
				delta = np.dot(self.layers[-i+1].weights.transpose(), delta) * currlay.activation(currlay.z, derivative = True)
				bias[-i] = delta
				weights[-i] = np.dot(delta, input_)
			return (bias, weights)

	'''
	Function Name: add

	Function Description:
	This function adds a layer to the neural network.
	'''
	def add(self, t_layer):
		self.layers.append(t_layer)

	'''
	Function Name: predict

	Function Description:
	Function that returns a prediction for an observation

	Function Parameters:
	-X: Instance(s) to predict for.
	'''
	def predict(self, X):
		pred_list = []
		total_obs = X.shape[0]
		for i in range(total_obs): # change to 0
			output = self.iterLayers(X[i,:])
			if self.layers[-1].units == 1:
				pred_list.append(float(output[0][0]))
			else:
				pred_list.append(self.getMax(output))
		return pred_list

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
	'''
	def mse(self, y, y_hat, derivative = False):
		if derivative:
			return 2 * (y_hat - y)
		else:
			return np.square(y - y_hat)

	'''
	Function Name: mutate

	Function Description:
	Function that randomly "mutates" the weights and biases.

	Function Parameters:
	-scalar: Number to scale how much we mutate the weights and biases.
	'''
	def mutate(self, scalar = 0.1):
		for layer in self.layers:
			# random.choice for what direction to mutate.
			layer.weights += np.random.rand(layer.units, layer.input_shape) * scalar * random.choice([-1, 1]))
			layer.bias += np.random.rand(layer.units, 1) * scalar * random.choice([-1, 1]))

	'''
	Function Name: crossover

	Function Description:
	Function that "crosses" over weights with another neural network.

	Later add in different types of crossover

	Function Parameter:
	-other: The other neural network with identical structure to cross over the weights.
	-type: The type of crossover to use. (not implemented yet, will be implemented at a later date)

	Function Requirements:
	- The other class needs to be of the same structure or this will not work.
	- Having an even number of layers is best but not required. (the last layer in odd will never be crossed over)
	'''
	def crossover(self, other, type = None):
		for i in range(0, len(self.layers), 2):
			if (i+1) < len(self.layers):
				self.layers[i].weights, other.layers[i+1].weights = other.layers[i+1].weights. self.layers[i].weights

	def __repr__(self):
		return "Artificial Neural Network"

	def __str__(self):
		return "Artificial Neural Network"

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
		self.delta = None # to be used in backropagation

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
		elif activation == "softmax":
			self.activation = self.softmax

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
		self.z = np.dot(self.weights, inputx.reshape(inputx.shape[0], 1)) + self.bias
		#print("Z", self.z)
		self.output = self.activation(self.z)
		return self.output

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
		self.weightsbp = None # bp for backpropagation.
		if iw == "random":
			self.weights = np.random.rand(self.units, self.input_shape)
			self.weightsbp = np.zeros((self.units, self.input_shape), dtype = float)
		elif iw == "random_s":
			self.weights = np.random.rand(self.units, self.input_shape) * 0.1
			self.weightsbp = np.zeros((self.units, self.input_shape), dtype = float)

	'''
	Function Name: reset

	Function Description:
	This function resets the values to zero when called. 
	This is to be called when we reach a new "batch" of data.
	'''
	def reset(self):
		self.biasbp.fill(0)
		self.weightsbp.fill(0)

	'''
        Function Name: updateWB

        Function Description:
        This function updates the weights based on the parameters.

	Function Parameters:
	-learning_rate: The value for the learning rate.
        '''
	def updateWB(self, learning_rate):
		w = self.weights
		#print("bp: ", self.weightsbp)
		#print("W: ", w)
		self.weights = self.weights - (self.weightsbp * learning_rate)
		#print("self.weights: ", self.weights)
		#print()
		#print()
		self.bias = self.bias - (self.biasbp * learning_rate)
		self.reset()

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
		self.biasbp = None # bp for backpropagation.
		if ib == "random":
			self.bias = np.random.rand(self.units, 1)
			self.biasbp = np.zeros((self.units, 1), dtype = float)
		elif ib == "random_s":
			self.bias = np.random.rand(self.units, 1) * 0.1
			self.biasbp = np.zeros((self.units, 1), dtype = float)

	'''
	Function Name: sigmoid

	Function Description:
	Implementation of the sigmoid activation function.

	Function Paramters:
	-x: value to "squish" between zero and one.
	-derivative: boolean value specifying what to return.
	'''
	def sigmoid(self, x, derivative = False):
		sig = 1.0 / (1 + np.exp(-x))
		if derivative:
			return sig * (1 - sig)
		else:
			return sig

	'''
	Function Name: relu

	Function Description:
	Implementation of the relu () activation function.

	Function Parameters:
	-x: if negative convert this value to 0, otherwise it is the same.

	note: has issues with "dead" relu.
	'''
	def relu(self, x, derivative = False):
		if derivative:
			return np.where(x > 0, 1, 0)
		else:
			return np.maximum(0, x)

	'''
	Function Name: softmax

	Function Description:
	Implementation of the softmax activation function.

	Function Parameters:
	-x The value to apply this activation function to
	'''
	def softmax(self, x, derivative = False):
		sm = np.exp(x) / np.sum(np.exp(x), axis = 0)
		if derivative:
			'''
			# https://themaverickmeerkat.com/2019-10-23-Softmax/ The previous link is where the derivitave version has been taken from.
			m,n = x.shape
			# t1 and t2 are tensors
			t1 = np.einsum("ij,ik->ijk", sm, sm)
			t2 = np.einsum("ij,jk->ijk", sm, no.eye(n,n))
			dersm = t2 - t1
			print("dersm", dersm)
			return dersm
			'''

			return sm * (1 - sm)
		else:
			return sm

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
	def elu(self, x, alpha, derivative = False):
		if derivative:
			if x >= 0:
				return 1
			else:
				return alpha * np.exp(x)
		else:
			if alpha < 0:
				raise Exception("Alpha should not be negative The value of alpha was: {}".format(alpha))
			else:
				if x >= 0:
					return x
				else:
					return alpha * (np.exp(x) - 1)

'''
def main():

	# Test that getOutput function in Dense is correct.
	#d = Dense(3, 6, "sigmoid", "random", "random")
	#print(d.bias.shape)
	#print("output", d.softmax(d.getOutput(np.array([0.07172362, 0.83726111, 0.11233911]))))
	#print(np.array([0.07172362, 0.83726111, 0.11233911]).reshape((3,1)))
	print(np.array([1,2,3]).reshape((3,1)) - np.array([1,1,1]).reshape((3,1)))
#) - (np.array([1,2,3]) * 5))

main()
'''
