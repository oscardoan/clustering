# Apply Spectral Clustering + KMean+ Agglomerative+ DBSCAN
# Author: Doan Tri Duc - 14520178
# Last Updated: 20/10/2017

import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import decimal
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_lfw_people

import training_data
from sklearn import metrics

data = training_data.load()
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
label_true = lfw_people.target
#SpectralClustering
graph_data = cosine_similarity(data)
y_spectral = SpectralClustering(n_clusters=7, eigen_solver='arpack', affinity='precomputed').fit_predict(graph_data)
reduced_data = PCA(n_components=2).fit_transform(data)
acc_spec = round (decimal.Decimal(metrics.adjusted_mutual_info_score(label_true,y_spectral)),2)


#DBSCAN
data_dbscan = scale(data)
data_dbscan = PCA(n_components=2).fit_transform(data_dbscan)
y_dbscan = DBSCAN(eps=1, min_samples=2).fit_predict(data_dbscan)
acc_dbscan = round (decimal.Decimal(metrics.adjusted_mutual_info_score(label_true,y_dbscan)),2)

#Agglomerative Clustering
data_agg = PCA(n_components=2).fit_transform(data)
y_agg = AgglomerativeClustering(n_clusters=7).fit_predict(data_agg)
acc_agg = round (decimal.Decimal(metrics.adjusted_mutual_info_score(label_true,y_agg)),2)

#KMEAN
X = PCA(n_components=2).fit_transform(data)
y_kmean = KMeans(init='k-means++', n_clusters=7).fit_predict(data)
acc_kmean = round (decimal.Decimal(metrics.adjusted_mutual_info_score(label_true,y_kmean)),2)


f, axarr = plt.subplots(2, 2)
f.suptitle("Clustering use LBP feature ", fontsize=16)
f.tight_layout()
f.subplots_adjust(top=0.88)

#Spectral Clustering Plot
axarr[0, 0].scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_spectral)
axarr[0, 0].set_title('Spectal Clustering- Accuracy: ' + str(acc_spec))
#DBSCAN Plot
axarr[0, 1].scatter(data_dbscan[:, 0], data_dbscan[:, 1], c=y_dbscan)
axarr[0, 1].set_title('DBSCAN- Accuracy: ' + str(acc_dbscan))

#Agglomerative Clustering Plot
axarr[1, 0].scatter(data_agg[:, 0], data_agg[:, 1], c=y_agg)
axarr[1, 0].set_title('Agglomerative Clustering- Accuracy: ' + str(acc_agg))


#Kmean Plot
axarr[1, 1].scatter(X[:, 0], X[:, 1], c=y_kmean)
axarr[1, 1].set_title('Kmean- Accuracy: ' + str(acc_kmean))

plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

plt.show()
