# Apply KMean
# Author: Doan Tri Duc - 14520178
# Last Updated: 20/10/2017

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hog_feature

data = hog_feature.load()
X = PCA(n_components=2).fit_transform(data)
y = KMeans(init='k-means++', n_clusters=7).fit_predict(data)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()