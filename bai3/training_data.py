import logging
from sklearn.datasets import fetch_lfw_people
import os
logging.basicConfig()
from skimage.feature import local_binary_pattern
import numpy as np

print os.path.dirname(os.path.abspath(__file__))
print os.path.isfile('./lbp_feature.npy')    # True  



# print os.path 

def load():
    if os.path.isfile('./lbp_feature.npy'):
        print "true"
        return np.load('lbp_feature.npy')
    else:
        print "false"
        lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        features = np.array([]).reshape(0,1850)
        for image in lfw_people.images: 
            lbp_feature = local_binary_pattern(image,P=8,R=0.5).flatten()
            print lbp_feature.shape
            features = np.append(features,[lbp_feature],axis=0)
        np.save(file='lbp_feature.npy',arr=features)
        return np.load('lbp_feature.npy')