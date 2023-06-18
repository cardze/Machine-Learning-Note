import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encode the target variable
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

# Set the number of hidden units to vary
hidden_units = range(10, 110, 10)

# Initialize a list to store the accuracies
accuracies = []

# Repeat the experiment 10 times
for _ in range(10):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Iterate over different hidden units
    for units in hidden_units:
        # Build the neural network model
        model = keras.Sequential([
            keras.layers.Dense(units, input_shape=(4,), activation='relu'),
            keras.layers.Dense(units, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=10, verbose=0)

        # Evaluate the model on the test set
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(accuracy)

# Calculate the average accuracy
average_accuracy = np.mean(accuracies)

# Print the average accuracy
print("Average accuracy:", average_accuracy*100, "%")