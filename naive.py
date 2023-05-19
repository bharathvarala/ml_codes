import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
data = pd.read_excel("buys_computer.xlsx")
# print(data)

num = LabelEncoder()
data["Age"] = num.fit_transform(data["Age"])
data["Income"] = num.fit_transform(data["Income"])
data["Student"] = num.fit_transform(data["Student"])
data["Credit_rating"] = num.fit_transform(data["Credit_rating"])
data["buys_computers"] = num.fit_transform(data["buys_computers"])

X = data.iloc[:,1:-1]
y = data.iloc[:,-1]
print(X,y)
X_train,X_test,y_train,y_split = train_test_split(X,y,test_size = 0.3)

clf = GaussianNB()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(f1_score(y_test,y_pred))

#accuracy
print(clf.score(X_test,y_test))
print(clf.predict([[2,1,0,1]]))
