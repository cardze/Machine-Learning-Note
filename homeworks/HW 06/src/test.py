import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def perceptron_learning(X, y, learning_rate, epochs):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    accuracy_list = []

    for _ in range(epochs):
        errors = 0

        for i in range(n_samples):
            x = X[i]
            y_true = y[i]
            y_pred = np.dot(weights, x) + bias
            update = learning_rate * (y_true) if y_true*y_pred <= 0 else 0
            weights += update * x
            bias += update
            errors += int(update != 0.0)

        accuracy = 1 - (errors / n_samples)
        accuracy_list.append(accuracy)

    return np.mean(accuracy_list)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Filter samples for classes "virginica" (2) and "versicolor" (1)
X_filtered = X[np.logical_or(y == 1, y == 2)]
y_filtered = y[np.logical_or(y == 1, y == 2)]

# Convert class labels to binary values (-1 for versicolor, 1 for virginica)
y_filtered = np.where(y_filtered == 1, -1, 1)

learning_rate_values = [0.01, 0.1, 0.5]  # Example learning rate values to test
num_epochs = 300  # Example number of training epochs


for learning_rate in learning_rate_values:
    accuracy_results = []
    for _ in range(10):
        # Split the dataset into training and testing sets (70% training, 30% testing)
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=0.3
        )

        accuracy = perceptron_learning(
            X_train, y_train, learning_rate, num_epochs
        )
        accuracy_results.append(accuracy)

    average_accuracy = np.mean(accuracy_results)
    print("Learning rate {0} \tAverage Accuracy:{1}".format(learning_rate, average_accuracy))
