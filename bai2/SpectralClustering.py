# Apply Spectral Clustering
# Author: Doan Tri Duc - 14520178
# Last Updated: 20/10/2017

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



X = metrics.pairwise.cosine_similarity(digits.data)


labels = spectral_clustering(X, n_clusters=10, eigen_solver='arpack')


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