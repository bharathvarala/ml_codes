import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_excel("data.xlsx")
print(data)

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

clf = svm.SVC(kernel = "linear")
# 
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(clf.predict([[5.2,3.4]]))
