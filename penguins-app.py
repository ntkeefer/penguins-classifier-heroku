# resource: https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

csv_url = 'https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv'
penguins = pd.read_csv(csv_url)

# Ordinal feature encoding

df = penguins.copy()
target = 'species'  # can switch with sex
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}


def target_encode(val):
    return target_mapper[val]


df['species'] = df['species'].apply(target_encode)

# Separating X and y
X = df.drop('species', axis=1)
y = df['species']

# Get rid of NANs and Infinities
X_new = np.nan_to_num(X)
y_new = np.nan_to_num(y)

# Build random forest model
clf = RandomForestClassifier()
clf.fit(X_new, y_new)

# Saving the model
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))
