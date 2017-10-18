import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import training_data
from sklearn.preprocessing import scale
data = training_data.load()
data = scale(data)
data = PCA(n_components=2).fit_transform(data)
y = DBSCAN(eps=0.6, min_samples=2).fit_predict(data)
plt.scatter(data[:, 0], data[:, 1], c=y)
plt.show()