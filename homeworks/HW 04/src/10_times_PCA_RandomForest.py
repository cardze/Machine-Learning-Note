import pandas as pd

from numpy import average
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cancer_data = pd.read_csv("../data/breast-cancer-wisconsin.data", 
                          names=[i for i in range(11)])
cancer_data = cancer_data.drop(cancer_data.columns[0], axis=1)
cancer_data = cancer_data.replace(to_replace='?', value=pd.NA)

## remove the row that miss the attribute
cancer_data.dropna(inplace = True) 
cancer_data = cancer_data.reset_index(drop=True)

# split the data
data_train, data_test, target_train, target_test = train_test_split(
    cancer_data[[i for i in range(1, 10)]], cancer_data[10], test_size=0.3
    )


# iterate from 1 to 9 and show the accuracy of it
for n_component in range(1,10):
    accurs = []
    for iter in range(10):
        # split the data
        data_train, data_test, target_train, target_test = train_test_split(
            cancer_data[[i for i in range(1, 10)]], cancer_data[10], test_size=0.3
        )
        model = PCA(n_components=n_component)
        data_train = model.fit_transform(data_train)
        data_test = model.transform(data_test)
        
        # random forest classifier
        classifier = RandomForestClassifier()
        classifier.fit(data_train, target_train)

        target_pred = classifier.predict(data_test)
        accurs.append(accuracy_score(target_test, target_pred))
    print("10 times accuracy = {0}  \tin choosing {1} components."
          .format(str(average(accurs)), n_component))
