from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier,AdaBoostClassifier,VotingClassifier 
from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score ,recall_score,f1_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
rfc = RandomForestClassifier(n_estimators = 100)
classifiers = {"dtc" : DecisionTreeClassifier(),
               "rfc" : RandomForestClassifier(n_estimators = 100),
               "bc" : BaggingClassifier(n_estimators = 100),
               "gbc" : GradientBoostingClassifier(n_estimators = 100),
               "abc" : AdaBoostClassifier(n_estimators = 100,learning_rate = 0.3),
               "vc" :VotingClassifier(estimators = [('rfc', rfc)],voting = "soft")}
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

metrics = {}

for i in classifiers:
    clf = classifiers[i]
    clf.fit(X_train,y_train)
    metrics[i] = []
    y_pred = clf.predict(X_test)
    metrics[i].append(["accuracy",accuracy_score(y_test,y_pred)])
    metrics[i].append(["precision",precision_score(y_test,y_pred,average = "micro")])
    metrics[i].append(["recall",recall_score(y_test,y_pred,average = "micro")])
    metrics[i].append(["f1_score",f1_score(y_test,y_pred,average = "micro")])

for i in metrics:
    print(i,metrics[i])
    print("\n")
