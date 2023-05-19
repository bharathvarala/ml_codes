#ID3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt

data = pd.read_csv("weather.nominal.csv")
print(data)
num = LabelEncoder()
data["outlook"] = num.fit_transform(data["outlook"])
data["temperature"] = num.fit_transform(data["temperature"])
data["humidity"] = num.fit_transform(data["humidity"])
data["windy"] = num.fit_transform(data["windy"])
data["play"] = num.fit_transform(data["play"])
print(data)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

# print(X)
# print(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

clf = DecisionTreeClassifier(criterion = "entropy",splitter = "best")
clf = clf.fit(X_train,y_train)
feature_names = X.columns
# clf.get_params()

# fig = plt.figure(figsize = (25,20))
_ = tree.plot_tree(clf,feature_names=feature_names,
                  class_names={0:"NO",1:"YES"},
                  filled = True,
                  fontsize=12)
