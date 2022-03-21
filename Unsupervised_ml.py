# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 14:14:48 2022

@author: TiaaUser
"""

""" Consider below machine learning apllication which implements K mean Algorithm for randomly generated dataset"""

import numpy as np
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt

def Marv():
    center_1 = np.array([1,1])
    
    center_2 = np.array([5,5])
    
    center_3 = np.array([8,1])
    
    #Generate random data and center it to the three centers
    
    data_1 = np.random.randn(7,2) + center_1
    
    print("Elements of first cluster with size" + str(len(data_1)))
    print(data_1)
    
    data_2 = np.random.randn(7,2) + center_2
    print("Elements of first cluster with size" + str(len(data_2)))
    print(data_2)
    
    data_3 = np.random.randn(7,2) + center_2
    print("Elements of first cluster with size" + str(len(data_3)))
    print(data_3)
    
    
    data = np.concatenate((data_1,data_1), axis = 0)
    print("Size of complete data set"+ str(len(data)))
    
    plt.scatter(data[:,0], data[:,1], s=7)
    plt.title('Input dataset')
    plt.show()
    
    k = 3
    
    n = data.shape[0]
    print('Total number of elements are', n)
    
    c= data.shape[1]
    print('Total number of features are', c)
    
    #Generate random centers, here we use sigma and means to ensuer it represent the whole data
    
    means = np.mean(data,axis=0)
    print("Value of means", means)
    
    # calculate standard 
    std = np.std(data,axis=0)
    print("Value of means", std)
    
    centers = np.random.randn(k,c)*std + means
    print("Random points are", centers)
    
    # plot the data and the centers generated as random
    plt.scatter(data[:,0], data[:,1],c='g', s=7)
    plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s = 150)
    plt.title("Input database with random centroid")
    plt.show()
    
    centers_old = np.zeros(centers.shape)
    centers_new = deepcopy(centers)
    
    print("Values of old centroids")
    print(centers_old)
    print("Values of old centroids")
    print(centers_new)
    
    data.shape
    clusters = np.zeros(n)
    distances = np.zeros((n,k))
    
    print("Values of old centroids")
    print(distances)
    
    error = np.linalg.norm(centers_new - centers_old)
    
    while error != 0:
    # Measure the distance to every center
        for i in range(k):
            distances[:,i] = np.linalg.norm(data-centers[i], axis = 1)
        
        # Assign all trainng data to closest center
        clusters = np.argmin(distances, axis = 1)
        centers_old = deepcopy(centers_new)
        
        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis = 0)
        error = np.linalg.norm(centers_new - centers_old)
   
        # end of while
        centers_new
       
        # plot the data and the centers generated as random
       
        plt.scatter(data[:,0], data[:,1],s =7)
        plt.scatter(centers_new[:,0], centers_new[:,1],marker='*', c = 'g', s =150)
        plt.title('Final data with Centroids')
        plt.show()
        
         
def main():
    print("Unsupervised Machine learning")
    print("Clustering using k Means Algorithm")
    Marv()
if __name__ == '__main__':
    main()    
    
    