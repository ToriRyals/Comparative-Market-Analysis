# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:50:09 2019

@author: bjvtc
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import csv
from sklearn.cluster import estimate_bandwidth


X = []
price = []
with open('prepared_dataset.csv') as PreparedData:
    csvReader = csv.reader(PreparedData)
    next(csvReader) #skips the header line
    for row in csvReader:
        X.append([int(row[6]), int(row[5]), int(row[3]),int(row[4])])
        price.append(int(row[6]))
arr = np.array(X)


colors = 10*["g","r","c","b","k"]


class Mean_Shift:
    def __init__(self, radius=estimate_bandwidth(arr)):
        self.radius = radius


    def fit(self, data):
        centroids = {}


        for i in range(len(data)):
            centroids[i] = data[i]
        
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset-centroid) < self.radius:
                        in_bandwidth.append(featureset)


                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))


            uniques = sorted(list(set(new_centroids)))


            prev_centroids = dict(centroids)


            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])


            optimized = True


            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
                
            if optimized:
                break


        self.centroids = centroids


fig = plt.figure(figsize=(12,10))
ax1 = fig.add_subplot(111, projection='3d')


from datetime import datetime
start_time = datetime.now()


clf = Mean_Shift()


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


clf.fit(arr)


centroids = clf.centroids
print(len(arr))
ax1.scatter(arr[:,0], arr[:,1], arr[:,2], s = 150, c = price, cmap = "hot")


for c in centroids:
    ax1.scatter(centroids[c][0], centroids[c][1], centroids[c][2], color='k', marker='*', s=300)


plt.show()

