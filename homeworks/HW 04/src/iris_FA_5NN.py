import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import FactorAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)

# preprocessing
X = dataset.drop('Class', axis = 1)
y = dataset['Class']
# 10 times iteration
for i in range(10):
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    fa = FactorAnalysis(n_components=3)
    X_train = fa.fit_transform(X_train)
    X_test = fa.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print('Accuracy ' + str(accuracy_score(y_test, y_pred)), str(i), "iteration")