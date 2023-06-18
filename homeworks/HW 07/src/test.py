import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Reshape and normalize the input data
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Define the modified LeNet-5 model
model = keras.Sequential([
    layers.Conv2D(6, kernel_size=5, strides=1, activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.Conv2D(16, kernel_size=5, strides=1, activation="relu"),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.Flatten(),
    layers.Dense(120, activation="relu"),
    layers.Dense(84, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model for 20 epochs and report average accuracy after 10 trials
num_trials = 10
num_epochs = 20
accuracy_scores = []

for _ in range(num_trials):
    model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    accuracy_scores.append(accuracy)

average_accuracy = sum(accuracy_scores) / num_trials
print("Average accuracy after 10 trials:", average_accuracy)