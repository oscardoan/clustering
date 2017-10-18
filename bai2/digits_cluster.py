import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

digits = load_digits()
data= digits.data
#SpectralClustering
graph_data = cosine_similarity(data)
y_spectral = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity='precomputed').fit_predict(graph_data)
reduced_data = PCA(n_components=2).fit_transform(data)

#DBSCAN
#data_dbscan = scale(data)
data_dbscan = PCA(n_components=2).fit_transform(data)
y_dbscan = DBSCAN(eps=0.6, min_samples=2).fit_predict(data_dbscan)

#Agglomerative Clustering
data_agg = PCA(n_components=2).fit_transform(data)
y_agg = AgglomerativeClustering(n_clusters=10).fit_predict(data_agg)

#KMEAN
X = PCA(n_components=2).fit_transform(data)
y_kmean = KMeans(init='k-means++', n_clusters=10).fit_predict(data)

f, axarr = plt.subplots(2, 2)
f.suptitle("Clustering in Digits Dataset ", fontsize=16)
f.tight_layout()
f.subplots_adjust(top=0.88)

#Spectral Clustering Plot
axarr[0, 0].scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_spectral)
axarr[0, 0].set_title('Spectal Clustering')
#DBSCAN Plot
axarr[0, 1].scatter(data_dbscan[:, 0], data_dbscan[:, 1], c=y_dbscan)
axarr[0, 1].set_title('DBSCAN')

#Agglomerative Clustering Plot
axarr[1, 0].scatter(data_agg[:, 0], data_agg[:, 1], c=y_agg)
axarr[1, 0].set_title('Agglomerative Clustering')


#Kmean Plot
axarr[1, 1].scatter(X[:, 0], X[:, 1], c=y_kmean)
axarr[1, 1].set_title('Kmean')

plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

plt.show()
