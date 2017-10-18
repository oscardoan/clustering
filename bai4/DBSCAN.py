import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import hog_feature
from sklearn.preprocessing import scale


data = hog_feature.load()
data = scale(data)
data = PCA(n_components=2).fit_transform(data)
y = DBSCAN(eps=0.6, min_samples=2).fit_predict(data)


plt.scatter(data[:, 0], data[:, 1], c=y)
plt.show()
