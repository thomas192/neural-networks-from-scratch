import numpy as np

# BACKPROPAGATION OF A LAYER OF NEURONS

# Passed-in gradient from the next layer
"""
dvalues = np.array([[1., 1., 1.]])
"""
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# 3 sets of weights - one set for each neuron
# 4 inputs, thus 4 weights
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases  # Dense layer
relu_outputs = np.maximum(0, layer_outputs)  # ReLU activation

# Backpropagation
# ReLU activation - simulates derivative with respect to input values
# from next layer passed to current layer during backpropagation
# ReLU derivative equals 1 if the input is greater than 0 and 0 otherwise
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

"""
# Sum weights related to the given input multiplied by the gradient related to the given neuron
# the partial derivative with respect to the input equals the related weight
dx0 = sum([weights[0][0]*dvalues[0][0], weights[0][1]*dvalues[0][1], weights[0][2]*dvalues[0][2]])
dx1 = sum([weights[1][0]*dvalues[0][0], weights[1][1]*dvalues[0][1], weights[1][2]*dvalues[0][2]])
dx2 = sum([weights[2][0]*dvalues[0][0], weights[2][1]*dvalues[0][1], weights[2][2]*dvalues[0][2]])
dx3 = sum([weights[3][0]*dvalues[0][0], weights[3][1]*dvalues[0][1], weights[3][2]*dvalues[0][2]])

dinputs = np.array([dx0, dx1, dx2, dx3])
"""
"""
# Sum weights related to the given input multiplied by the gradient related to the given neuron
# the partial derivative with respect to the input equals the related weight
dx0 = sum(weights[0]*dvalues[0])
dx1 = sum(weights[1]*dvalues[0])
dx2 = sum(weights[2]*dvalues[0])
dx3 = sum(weights[3]*dvalues[0])

dinputs = np.array([dx0, dx1, dx2, dx3])
"""

# Dense layer
# Sum weights of given input
# and multiply by the passed-in gradient for this neuron
# the partial derivative with respect to the input equals the related weight
dinputs = np.dot(drelu, weights.T)
print("\nGradients with respect to inputs:")
print(dinputs)

# Sum weights of given input
# and multiply by the passed-in gradient for this neuron
# the derivative with respect to the weights equals inputs
dweights = np.dot(inputs.T, drelu)
print("\nGradients with respect to weights:")
print(dweights)

# dbiases - sum values, do this over samples (first axis)
# keepdims since this by default will produce a plain list (chapter 4)
dbiases = np.sum(drelu, axis=0, keepdims=True)
print("\nGradients with respect to biases:")
print(dbiases)
