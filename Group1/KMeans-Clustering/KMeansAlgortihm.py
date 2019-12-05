# imports the csv file and each row of the file is returned as an array
import csv
import numpy as np # carrys out computations
from sklearn.cluster import KMeans #inlcudes KMeans functionality
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D #used to plot 3D graphs
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt #2D plotting library
import pandas as pd
from datetime import datetime

#initialz arrays to hold data values
street = []
city = []
zipcode = []
beds = []
baths = []
sq__ft = []
price = []

with open('prepared_dataset.csv') as PreparedData:
    csvReader = csv.reader(PreparedData)
    next(csvReader) #skips the header line
    for row in csvReader:
        street.append(row[0]) #appends data to above arrays
        city.append(row[1])
        zipcode.append(row[2])
        beds.append(row[3])
        baths.append(row[4])
        sq__ft.append(row[5])
        price.append(row[6])


		
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


#Changes data values from string to int
x = list(map(int, price))
y=list(map(int, sq__ft))
z=list(map(int, beds))
c=list(map(int, baths))


#### creates original graph before KMeans Algorithm
img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
ax.set_xlabel("price", color = 'red');
ax.set_ylabel("sq__ft", color = 'orange');
ax.set_zlabel("bed", color = 'yellow');
plt.title("price, sq__ft, beds, baths");

out_png2 = "KMeansOriginalgraph.png"
plt.savefig(out_png2, dpi=150)
plt.show()


#### Iterate KMeans Algorithm
start_time = datetime.now()

X=np.matrix(list(zip(x,y,z,c))) #zips csv colums, creates one list & returns a matrix
kmeans = KMeans(n_clusters=4).fit(X) #runs through KMeans steps (built into sklearn library)

print ("The coordinate of cluster centers are: ", kmeans.cluster_centers_) #points for center of clusters
print (" The number of iterations ran is: ", kmeans.n_iter_) #number of iterations


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time)) #time it takes algorithm to run


#### Final Graph after algorithm

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')


img = ax.scatter(X[:,0],X[:,1], c=kmeans.labels_, s = 1, cmap='rainbow')
img1 = ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, color='black')

plt.show()	
	
ax.set_xlabel("price");
ax.set_ylabel("sq__ft");
ax.set_zlabel("bed");

plt.title("price, sq__ft, beds, baths");

out_png2 = "KMeansFinalgraph.png"
plt.savefig(out_png2, dpi=75)
plt.show()



## print what cluster each point is in 
create_file= open("ClusterGroup.csv","w")
with open('ClusterGroup.csv') as OutputData:

	create_file.write("Cluster Group"'\n')
	vertical=('\n'.join(map(str,kmeans.labels_)))
	create_file.write(vertical)
create_file.close()


# adds clusters group to original file
with open('prepared_dataset.csv', 'r') as t1, open('ClusterGroup.csv', 'r') as t2, open('FinalData.csv', 'w') as output:
     r1 = csv.reader(t1)
     r2 = csv.reader(t2)
     w = csv.writer(output)
     for a, b in zip(r1, r2):
             w.writerow(a + b)
create_file.close()




