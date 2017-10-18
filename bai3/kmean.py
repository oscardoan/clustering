#Kmean
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import training_data

data = training_data.load()
X = PCA(n_components=2).fit_transform(data)
y = KMeans(init='k-means++', n_clusters=10).fit_predict(data)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()