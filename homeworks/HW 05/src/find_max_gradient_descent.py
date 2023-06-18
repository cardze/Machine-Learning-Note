import numpy as np
import math

def cost_function(x, y):
    return x + y - 100 * (x**2 + y**2 - 1)**2

def gradient(x, y):
    dx = 1 - 400 * x * round(math.pow(x,2) + math.pow(y,2) - 1, 5)
    dy = 1 - 400 * y * round(math.pow(x,2) + math.pow(y,2) - 1, 5)
    # print(dx, dy)
    return np.array([dx, dy])

def gradient_descent(learning_rate, num_iterations):
    # Initial values
    x = 0.5
    y = 0.5

    # Perform gradient descent
    for i in range(num_iterations):
        grad = gradient(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]

    return x, y

# Set hyperparameters
learning_rate = 0.0001
num_iterations = 100

# Run gradient descent
x_optimized, y_optimized = gradient_descent(learning_rate, num_iterations)

# Print the optimized values
print("Optimized values:")
print("x =", x_optimized)
print("y =", y_optimized)
# print(gradient(1.0, 1.0))