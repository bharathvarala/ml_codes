#logistic regression

import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

data = pd.read_excel("buys_computer.xlsx")


num = LabelEncoder()
data["Age"] = num.fit_transform(data["Age"])
data["Income"] = num.fit_transform(data["Income"])
data["Student"] = num.fit_transform(data["Student"])
data["Credit_rating"] = num.fit_transform(data["Credit_rating"])
data["buys_computers"] = num.fit_transform(data["buys_computers"])
X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# print(X_train,y_train)
clf = LogisticRegression(random_state=0)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
