# Generate traning data
# Author: Doan Tri Duc - 14520178
# Last Updated: 20/10/2017

import logging
from sklearn.datasets import fetch_lfw_people
import os
logging.basicConfig()
from skimage.feature import local_binary_pattern
import numpy as np
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize

print os.path.dirname(os.path.abspath(__file__))
print os.path.isfile('./lbp_feature.npy')    # True  



# print os.path 

def load():
    if os.path.isfile('./lbp_feature.npy'):
        print "true"
        return np.load('lbp_feature.npy')
    else:

        lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        np.save(file='target.npy', arr=lfw_people.target)
        X = []
        for i in range(len(lfw_people.images)):
            lbt_image = local_binary_pattern(lfw_people.images[i], P=24, R=3, method='uniform')
            (lbt_hist,_) = np.histogram(lbt_image.ravel(), bins=int(lbt_image.max() + 1), range=(0, 24 + 2))
            X.append(lbt_hist)
        X = np.array(X)
        np.save(file='lbp_feature.npy',arr=X)
        return np.load('lbp_feature.npy')

if __name__ == "__main__":
    load()