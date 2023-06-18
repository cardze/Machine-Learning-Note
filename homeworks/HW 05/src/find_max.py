from scipy.optimize import minimize

def cost_function(x):
    x, y = x
    return -(x + y - 100 * (x**2 + y**2 - 1)**2)

# Initial guess
x0 = [0, 0]

# Optimization
result = minimize(cost_function, x0, method='Nelder-Mead')

# Extracting the optimized values
x_optimized, y_optimized = result.x

# Print the optimized values
print("Optimized values:")
print("x =", x_optimized)
print("y =", y_optimized)
