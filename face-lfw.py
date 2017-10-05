
import logging
from sklearn.datasets import fetch_lfw_people
logging.basicConfig()

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

