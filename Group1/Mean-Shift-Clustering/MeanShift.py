import csv
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime


url = 'prepared_dataset.csv'
X = []
price = []
with open('prepared_dataset.csv') as PreparedData:
    csvReader = csv.reader(PreparedData)
    next(csvReader) #skips the header line
    for row in csvReader:
        X.append([int(float(row[0])), int(float(row[3])), int(float(row[1]))]) ##
        price.append(int(float(row[2])))
arr = np.array(X)

#### Iterate KMeans Algorithm
start_time = datetime.now()
ms = MeanShift().fit(arr)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time)) #time it takes algorithm to run

cluster_centers = ms.cluster_centers_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("bath");
ax.set_ylabel("bed");
ax.set_zlabel("sq_ft");
plt.title("price, sq_ft, beds, baths");
ax.scatter(arr[:,0], arr[:,1], arr[:,2], s = 150, c = price, cmap = "hot", marker='o')
ax.scatter(cluster_centers[:,0], cluster_centers[:,1], cluster_centers[:,2], marker='o', color='black', s=500, linewidth=10, zorder=10)
plt.show()

print(cluster_centers)
