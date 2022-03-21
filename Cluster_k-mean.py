# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 13:40:48 2022

@author: TiaaUser
"""

# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# importing the iris dataset with pandas

path = "iris.csv"
dataset = pd.read_csv(path)

x = dataset.iloc[:,[0,1,2,3]].values

# Finding the potimum number of cluster for k=mean classification

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init = 'k-means++', max_iter= 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# plotting the results onto a line graph, allowing us to observes"The elbow"

plt.plot(range(1,11),wcss)
plt.title("The elbow method") 
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#Applying Kmeans to the dataset / creating the kmeans classifier

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter= 300, n_init=10, random_state=0)

y_means = kmeans.fit_predict(x)

# Visualising the clusters

plt.scatter(x[y_means == 0,0], x[y_means ==0,1], s = 100, c ='red', label = 'Iris-setosa')   

plt.scatter(x[y_means == 1,0], x[y_means ==1,1], s = 100, c ='blue', label = 'Iris-versicolour')  

plt.scatter(x[y_means == 2,0], x[y_means ==2,1], s = 100, c ='green', label = 'Iris-virginica')  

# Plotting the centroidss of the clusters

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=100, c='yellow', label = 'Centroids')

plt.legend()

plt.show()
