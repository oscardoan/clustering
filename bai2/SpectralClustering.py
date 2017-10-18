#MSSV: 14520178 
# special clustering

print(__doc__)

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import spectral_clustering

np.random.seed(42)

digits = load_digits()
#data = scale(digits.data)


X = metrics.pairwise.cosine_similarity(digits.data)
#X = np.nan_to_num(X)
#a = np.asarray(X)
#print a.shape()
#print typeof(X)
#ar = np.array(X)
#X= X + 0.01

labels = spectral_clustering(X, n_clusters=10, eigen_solver='arpack')
#label_im = -np.ones(mask.shape)
#label_im[mask] = labels

#plt.matshow(img)
#plt.matshow(label_im)

# #############################################################################
# 2 circles
# img = circle1 + circle2
# mask = img.astype(bool)
# img = img.astype(float)

# img += 1 + 0.2 * np.random.randn(*img.shape)

# graph = image.img_to_graph(img, mask=mask)
# graph.data = np.exp(-graph.data / graph.data.std())

# labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')
# label_im = -np.ones(mask.shape)
# label_im[mask] = labels

# plt.matshow(img)
# plt.matshow(label_im)

# plt.show()

print("Result")
df = pd.DataFrame({'Labels':labels,'Truth labels':digits.target})
ct = pd.crosstab(df['Labels'],df['Truth labels'])
print(ct)
print("-----------------------------------------------------------------")

# visualize 
pca_converter = PCA(n_components = 2) 
# convert digits data to 2D points 
data_2d = pca_converter.fit_transform(digits.data) 
plt.scatter(data_2d[:,0], data_2d[:,1], c=labels)
plt.show()