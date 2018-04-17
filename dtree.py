import pandas as pd
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


"""
Script for training decision tree.
"""

# Get data
df = pd.read_csv('test_data.csv')
df = df.fillna(0)
df_x = df[df.columns[:len(df.columns) - 1]]
df_y = df[df.columns[-1]]
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=.1)


# Get classifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
joblib.dump(clf, 'clf.pkl')



