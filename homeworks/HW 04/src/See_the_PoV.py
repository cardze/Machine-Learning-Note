import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# model
model = PCA(
    n_components=9
)

# fit the data
model.fit_transform(cancer_data)

# show the PoV
explain = model.explained_variance_ratio_.cumsum()
for i in range(len(explain)):
    print("{0} component's PoV : {1} {2}".format(i+1, explain[i], "O" if explain[i]>0.9 else "X"))