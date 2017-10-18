from matplotlib import pyplot as plt
import hog_feature
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


data = hog_feature.load()
data = PCA(n_components=2).fit_transform(data)
y = AgglomerativeClustering(n_clusters=10).fit_predict(data)
plt.scatter(data[:, 0], data[:, 1], c=y)
plt.show()