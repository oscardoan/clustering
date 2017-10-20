# Apply Spectral Clustering
# Author: Doan Tri Duc - 14520178
# Last Updated: 20/10/2017

import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import hog_feature

data = hog_feature.load()
graph_data = cosine_similarity(data)
y = SpectralClustering(n_clusters=7, eigen_solver='arpack', affinity='precomputed').fit_predict(graph_data)
reduced_data = PCA(n_components=2).fit_transform(data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1],s=80, c=y)
plt.show()