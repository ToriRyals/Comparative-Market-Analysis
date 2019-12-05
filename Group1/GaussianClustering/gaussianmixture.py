# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:09:36 2019

@author: bjvtc
"""

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('.\prepared_dataset.csv')
my_data=data[['sq__ft','beds','price']].values

X = my_data
#Gaussian Mixture Model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
proba_lists = gmm.predict_proba(X)
#Plotting
colored_arrays = np.matrix(proba_lists)
colored_tuples = [tuple(i.tolist()[0]) for i in colored_arrays]
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 0], X[:, 1], X[:, 2],
          c=colored_tuples, edgecolor="k", s=50)
ax.set_xlabel("sq_ft")
ax.set_ylabel("beds")
ax.set_zlabel("price")
plt.title("Gaussian Mixture Model", fontsize=14)
