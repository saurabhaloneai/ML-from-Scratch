import numpy as np

def step_function(x):
    return 1 if x > 0 else 0

def perceptron_learning(x, y, learning_rate=0.1, epochs=1000):
    # Initialize weights and bias
    w = np.zeros(len(x[0]))
    theta = 0.0
    
    for epoch in range(epochs):
        errors = 0
        for i in range(len(x)):
            # Calculate the predicted output
            y_pred = step_function(np.dot(x[i], w) - theta)
            
            # Update weights and bias using gradient descent
            w += learning_rate * (y[i] - y_pred) * x[i]
            theta -= learning_rate * (y[i] - y_pred)
            
            errors += int(y[i] != y_pred)
        
        # Print the updates during each epoch
        print(f"Epoch {epoch+1}/{epochs} - Errors: {errors} - Weights: {w} - Theta: {theta}")
        
        # Stop if all examples are classified correctly
        if errors == 0:
            break
    
    return w, theta

# Training data for AND gate
x_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Find optimal weights and bias for AND gate
optimal_w_and, optimal_theta_and = perceptron_learning(x_and, y_and)

# Training data for OR gate
x_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

# Find optimal weights and bias for OR gate
optimal_w_or, optimal_theta_or = perceptron_learning(x_or, y_or)

# Training data for NOT gate
x_not = np.array([[0], [1]])
y_not = np.array([1, 0])

# Find optimal weights and bias for NOT gate
optimal_w_not, optimal_theta_not = perceptron_learning(x_not, y_not)

# Test the gates with the optimal parameters
def gate_output(x, w, theta):
    result = step_function(np.dot(x, w) - theta)
    return result

# Test the AND gate
print("\nOptimal Weights (AND):", optimal_w_and)
print("Optimal Theta (AND):", optimal_theta_and)
print("AND Gate Outputs:")
print(gate_output([0, 0], optimal_w_and, optimal_theta_and))
print(gate_output([0, 1], optimal_w_and, optimal_theta_and))
print(gate_output([1, 0], optimal_w_and, optimal_theta_and))
print(gate_output([1, 1], optimal_w_and, optimal_theta_and))

# Test the OR gate
print("\nOptimal Weights (OR):", optimal_w_or)
print("Optimal Theta (OR):", optimal_theta_or)
print("OR Gate Outputs:")
print(gate_output([0, 0], optimal_w_or, optimal_theta_or))
print(gate_output([0, 1], optimal_w_or, optimal_theta_or))
print(gate_output([1, 0], optimal_w_or, optimal_theta_or))
print(gate_output([1, 1], optimal_w_or, optimal_theta_or))

# Test the NOT gate
print("\nOptimal Weights (NOT):", optimal_w_not)
print("Optimal Theta (NOT):", optimal_theta_not)
print("NOT Gate Outputs:")
print(gate_output([0], optimal_w_not, optimal_theta_not))
print(gate_output([1], optimal_w_not, optimal_theta_not))
