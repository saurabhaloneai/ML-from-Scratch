#NAND 
def nand_gate(A, B):
    return not (A and B)

# OR 
def or_gate(A, B):
    return A or B

# AND 
def and_gate(A, B):
    return A and B

# XOR 
def xor_gate(A, B):
    output_nand1 = nand_gate(A, B)
    output_nand2 = nand_gate(A, output_nand1)
    output_nand3 = nand_gate(B, output_nand1)
    output_or = or_gate(output_nand2, output_nand3)
    final_result = and_gate(output_nand1, output_or)
    return final_result


for A in [0, 1]:
    for B in [0, 1]:
        result = xor_gate(A, B)
        print(f"XOR({A}, {B}) = {result}")

# Perceptron
class Perceptron:
    def __init__(self, input_size, learning_rate):
        self.weights = [0] * input_size
        self.bias = 0
        self.learning_rate = learning_rate

    def predict(self, inputs):
        activation = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return 1 if activation >= 0 else 0

    def update_weights(self, inputs, target):
        prediction = self.predict(inputs)
        error = target - prediction
        self.weights = [w + self.learning_rate * error * x for w, x in zip(self.weights, inputs)]
        self.bias += self.learning_rate * error


perceptron = Perceptron(input_size=2, learning_rate=1.5)

# Training 
training_data = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]

for epoch in range(20000):
    for inputs, target in training_data:
        perceptron.update_weights(inputs, target)

# Test
for inputs, target in training_data:
    prediction = perceptron.predict(inputs)
    print(f"Perceptron XOR({inputs}) = {prediction}")
