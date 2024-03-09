import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_mlp(x, y, hidden_neurons, epochs=10000, learning_rate=0.1):
    input_neurons = x.shape[1]
    output_neurons = 1

    
    np.random.seed(42)
    weights_input_hidden = np.random.rand(input_neurons, hidden_neurons)
    weights_hidden_output = np.random.rand(hidden_neurons, output_neurons)
    bias_hidden = np.zeros((1, hidden_neurons))
    bias_output = np.zeros((1, output_neurons))

    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(x, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_layer_input)

        # Backpropagation
        error = y - predicted_output
        output_delta = error * sigmoid_derivative(predicted_output)

        hidden_layer_error = output_delta.dot(weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

        # Update weights and biases
        weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
        bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        weights_input_hidden += x.T.dot(hidden_layer_delta) * learning_rate
        bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

def test_mlp(x, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    return np.round(predicted_output)

# XOR gate training data
x_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Train the XOR model
weights_input_hidden_xor, weights_hidden_output_xor, bias_hidden_xor, bias_output_xor = train_mlp(x_xor, y_xor, hidden_neurons=2)

# XNOR gate training data
x_xnor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xnor = np.array([[1], [0], [0], [1]])

# Train the XNOR model
weights_input_hidden_xnor, weights_hidden_output_xnor, bias_hidden_xnor, bias_output_xnor = train_mlp(x_xnor, y_xnor, hidden_neurons=2)

# Test the XOR model
predictions_xor = test_mlp(x_xor, weights_input_hidden_xor, weights_hidden_output_xor, bias_hidden_xor, bias_output_xor)

print("\nXOR Gate Weights (Input to Hidden):")
print(weights_input_hidden_xor)
print("XOR Gate Weights (Hidden to Output):")
print(weights_hidden_output_xor)
print("XOR Gate Bias (Hidden):")
print(bias_hidden_xor)
print("XOR Gate Bias (Output):")
print(bias_output_xor)
print("XOR Gate Predictions:")
print(predictions_xor.flatten())
print("XOR Gate True Outputs:")
print(y_xor.flatten())

# Test the XNOR model
predictions_xnor = test_mlp(x_xnor, weights_input_hidden_xnor, weights_hidden_output_xnor, bias_hidden_xnor, bias_output_xnor)

print("\nXNOR Gate Weights (Input to Hidden):")
print(weights_input_hidden_xnor)
print("XNOR Gate Weights (Hidden to Output):")
print(weights_hidden_output_xnor)
print("XNOR Gate Bias (Hidden):")
print(bias_hidden_xnor)
print("XNOR Gate Bias (Output):")
print(bias_output_xnor)
print("XNOR Gate Predictions:")
print(predictions_xnor.flatten())
print("XNOR Gate True Outputs:")
print(y_xnor.flatten())
