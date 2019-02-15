# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 12:31:20 2019

@author: KEARS4
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris

knn = KNeighborsClassifier(n_neighbors= 6)
knn.fit(iris.data, iris.target)
