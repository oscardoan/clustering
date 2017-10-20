# Genarate  HOG Feature
# Author: Doan Tri Duc - 14520178
# Last Updated: 20/10/2017

import logging
from sklearn.datasets import fetch_lfw_people
import os
logging.basicConfig()
from skimage.feature import local_binary_pattern
from skimage.feature import hog
import numpy as np
from skimage.feature import daisy
from skimage import img_as_float
print os.path.dirname(os.path.abspath(__file__))
print os.path.isfile('./sift_feature.npy')    # True  



# print os.path 

def load():
    if os.path.isfile('./hog_feature.npy'):
        return np.load('hog_feature.npy')
    else:
        lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        features = np.array([]).reshape(0,648)
        for img in lfw_people.images: 
            feature = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3))
            features = np.append(features,[feature],axis=0)
        np.save(file='hog_feature.npy',arr=features)
        return np.load('hog_feature.npy')