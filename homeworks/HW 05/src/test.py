import numpy as np


def gradient(x, y):
    dx = 1 - 400 * x * (x**2 + y**2 - 1)
    dy = 1 - 400 * y * (x**2 + y**2 - 1)
    print(dx, dy)
    return np.array([dx, dy])

def gradient_ascent(learning_rate, num_iterations):
    # Initial values
    x = 1.0
    y = 1.0

    # Perform gradient ascent
    for i in range(num_iterations):
        grad = gradient(x, y)
        x += learning_rate * grad[0]
        y += learning_rate * grad[1]
        # print(grad[0], grad[1])

    return x, y

# Set hyperparameters
learning_rate = 0.0005
num_iterations = 10

# Run gradient descent
x_optimized, y_optimized = gradient_ascent(learning_rate, num_iterations)

# Print the optimized values
print("Optimized values:")
print("x =", x_optimized)
print("y =", y_optimized)
