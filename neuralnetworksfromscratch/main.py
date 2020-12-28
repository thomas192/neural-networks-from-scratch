import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()


# Dense layer
class LayerDense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU activation
class ActivationReLU:
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)


# Softmax activation
class ActivationSoftmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = LayerDense(2, 3)

# Create ReLU activation (to be used with Dense layer)
activationRelu = ActivationReLU()

# Create second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values
dense2 = LayerDense(3, 3)

# Create Softmax activation (to be used with Dense layer)
activationSoftmax = ActivationSoftmax()

# Make a forward pass of the training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activationRelu.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activationRelu.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
activationSoftmax.forward(dense2.output)

print(activationSoftmax.output[:5])
