# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:45:11 2020

@author: HP
"""

def distanceNotZero(D,index,trainSize):
   for j in range(trainSize):
        if(D[index][j]==0):
            return False
   return True

def findCluster(ubest):
    Cluster = list()
    for j in range(len(ubest[0])):
        maxim = 0
        for i in range(len(ubest)):
           if(ubest[i][j] > maxim):
               maxim = ubest[i][j]
               cluster = i
        Cluster.append(cluster + 1)
    return Cluster 
with open('centroidfile.txt') as f:
    #w, h = [float(x) for x in next(f).split()] 
    V = []
    for line in f: 
        V.append([float(x) for x in line.split()])

with open('testfile.txt') as f:
    #w, h = [float(x) for x in next(f).split()] 
    Z = []
    for line in f: 
        Z.append([float(x) for x in line.split()])

import numpy as np
c = len(V)
no_of_features = len(Z[0])
test_size = len(Z)
U = [[0 for col in range(test_size)] for row in range(c)]
m = 2
Z = [np.array(i) for i in Z]
V = [np.array(i) for i in V]
A = np.identity(no_of_features,dtype = int)
D = list()
for i in range(c):
    d = list()
    for j in range(test_size):
        inp = np.dot(np.transpose((Z[j] - V[i])),A)
        inp = np.dot(inp,Z[j] - V[i])
        d.append(np.sqrt(inp))
    D.append(d)
for i in range(c):
    for j in range(test_size):
        if(distanceNotZero(D,i,test_size)):
            sum1 = 0
            for k in range(c):
                sum1 = sum1 + (D[i][j]/D[k][j])**(2/(m-1))
            U[i][j] = 1/sum1
            
        else:
            print('Entering')
            U[i][j] = 1
            for k in range(c):
                if(i!=k):
                    U[k][j] = 0

import matplotlib.pyplot as plt
sampleCluster = findCluster(U)
Z = np.array(Z)
plt.scatter(Z[:,0],Z[:,1],c=sampleCluster, cmap=plt.cm.Paired)
plt.savefig('Clusterplot')
file = open("ClusterFile.txt","w+")
file.write(" x    y    Cluster\n" )
for i in range(test_size):
    for j in range(no_of_features):
            file.write(str("{:.2f}".format(Z[i][j])))
            file.write(" ")
    file.write("    ")
    file.write(str(sampleCluster[i]))
    file.write("\n")

file.close()
