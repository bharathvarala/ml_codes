from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

iris = load_iris()
X = iris.data

sc = StandardScaler()
X_std = sc.fit_transform(X)

clf1 = KMeans(n_clusters = 3,n_init = 10,random_state = 42)
label1 = clf1.fit_predict(X_std)
clf2 = AgglomerativeClustering(n_clusters = 3)
label2 = clf2.fit_predict(X_std)
clf3 = DBSCAN(eps = 0.6, min_samples = 3)
label3 = clf3.fit_predict(X_std)


print(silhouette_score(X_std,label1))
print(silhouette_score(X_std,label2))
print(silhouette_score(X_std,label3))


fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,4))
ax1.scatter(X[:,0],X[:,1],c = label1)
ax2.scatter(X[:,0],X[:,1],c = label2)
ax3.scatter(X[:,0],X[:,1],c = label3)
plt.show()
