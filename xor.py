import numpy as np

# Perceptron class with gradient descent
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size) * 2 - 1  # Initialize weights between -1 and 1
        self.bias = np.random.rand() * 2 - 1  # Initialize bias between -1 and 1
        self.learning_rate = 0.1

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        return self.activation(np.dot(inputs, self.weights) + self.bias)

    def update_weights(self, inputs, error):
        gradient_weights = self.learning_rate * error * inputs
        gradient_bias = self.learning_rate * error
        self.weights += gradient_weights
        self.bias += gradient_bias

# NAND gate
nand_gate = Perceptron(2)

# OR gate
or_gate = Perceptron(2)

# AND gate
and_gate = Perceptron(2)

# Training NAND gate
nand_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nand_labels = np.array([1, 1, 1, 0])

for epoch in range(100):  # Increase the number of epochs
    for i in range(len(nand_data)):
        prediction = nand_gate.predict(nand_data[i])
        error = nand_labels[i] - prediction
        nand_gate.update_weights(nand_data[i], error)

# Training OR gate
or_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_labels = np.array([0, 1, 1, 1])

for epoch in range(10000):  # Increase the number of epochs
    for i in range(len(or_data)):
        prediction = or_gate.predict(or_data[i])
        error = or_labels[i] - prediction
        or_gate.update_weights(or_data[i], error)

# Combine NAND and OR gates to create XOR gate
xor_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

for i in range(len(xor_data)):
    nand_output = nand_gate.predict(xor_data[i])
    or_output = or_gate.predict(xor_data[i])
    and_input = np.array([nand_output, or_output])
    
    # Training AND gate
    and_labels = np.array([0, 1, 1, 0])
    and_output = and_gate.predict(and_input)
    and_error = and_labels[i] - and_output
    and_gate.update_weights(and_input, and_error)

# Testing XOR gate
for i in range(len(xor_data)):
    nand_output = nand_gate.predict(xor_data[i])
    or_output = or_gate.predict(xor_data[i])
    and_input = np.array([nand_output, or_output])
    
    xor_output = and_gate.predict(and_input)
    print(f"Input: {xor_data[i]}, Output: {xor_output}")
