import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cancer_data = pd.read_csv("../data/breast-cancer-wisconsin.data", 
                          names=[i for i in range(11)])
cancer_data = cancer_data.drop(cancer_data.columns[0], axis=1)
cancer_data = cancer_data.replace(to_replace='?', value=pd.NA)

## remove the row that miss the attribute
cancer_data.dropna(inplace = True) 
cancer_data = cancer_data.reset_index(drop=True)

# split the datasets
data_train, data_test, target_train, target_test = train_test_split(
    cancer_data[[i for i in range(1, 10)]], cancer_data[10], test_size=0.2
    )
data_train, data_valid, target_train, target_valid = train_test_split(
    data_train, target_train, test_size=0.25
    )

# Standarize
sc = StandardScaler()

data_train = sc.fit_transform(data_train)
data_test = sc.transform(data_test)

# establish model
model = LinearDiscriminantAnalysis()
model.fit(data_train, target_train)
target_pred = model.predict(data_test)

# result
cm = confusion_matrix(target_test, target_pred)
print(cm)
print("Accuracy = {0}".format(str(accuracy_score(target_test, target_pred))))
