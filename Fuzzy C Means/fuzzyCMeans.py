# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:37:12 2020

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:29:28 2020

@author: HP
"""
#not visit the sample again once d[i][j] is zero

import pandas as pd
import random
import numpy as np

dataset = pd.read_excel('Data Sets.xlsx','Data Set 2',header = None)
no_of_samples = len(dataset) 
C = [2,3,4,5,6,7,8,9,10,11]
m = 2
no_of_features = 2

Z_data = dataset
Z_data = Z_data.to_numpy() 
Z = list()
Z_test = list()

for i in range(no_of_samples):
    if(i<600):
        if((i+1)%5 != 0):
            Z.append(Z_data[i])
        else:
            Z_test.append(Z_data[i])
    else:
        Z_test.append(Z_data[i])

itr = 0
train_size = len(Z)
test_size = len(Z_test)
Z = np.array(Z)
Z_test = np.array(Z_test)
def column_sum(lst): 
     return np.sum(lst,axis=0).tolist()
def sumLessThanOne(U,i,r):
    sum_List = column_sum(U)
    if(sum_List[i] + r>=1):
        return True
    else:
        return False

def distanceNotZero(D,index,trainSize):
   for j in range(trainSize):
        if(D[index][j]==0):
            return False
   return True 

def objective(d,mu):
    J = 0
    for i in range(c):
        for j in range(len(d[i])):
           J = J + ((mu[i][j])**m)*( d[i][j] )**2
    return J
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

def findMin(ratioMatrix):
    minim = 999
    for i in range(len(ratioMatrix)):
        if(ratioMatrix[i] < minim and ratioMatrix[i] >0.009):
            minim = ratioMatrix[i]
            ind = i
    return ind
obj = list()
UofClusters = list()
iterations = list()    
VofClusters = list()

for c in C:
    
    
    U = np.random.rand(c,train_size)
    U /= np.sum(U, axis=0)
    U =  U.tolist()
    itr = 0
    while True:
        Uprev = np.array(U)
        itr = itr +1
        V = list()
        
        for i in range(c):
             Uarrsqr = np.square(np.array(U[i]))
             nr = sum(Z * Uarrsqr[:, np.newaxis])
             dr = sum(Uarrsqr)
             V.append(list(nr/dr))

        V = np.array(V)
        A = np.identity(no_of_features,dtype = int)
        D = list()
        for i in range(c):
            d = list()
            for j in range(train_size):
                inp = np.dot(np.transpose((Z[j] - V[i])),A)
                inp = np.dot(inp,Z[j] - V[i])
                d.append(np.sqrt(inp))
            D.append(d)
        for i in range(c):
            for j in range(train_size):
                if(distanceNotZero(D,i,train_size)):
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

        Ua1 =Uprev
        Ua2 = np.array(U)
        diff = (Ua1 - Ua2)
        diff = abs(diff)
        Umax = np.amax(diff)      
       
        print(itr)
        if(abs(Umax) < 0.01):
            break
    objVal = objective(D,U)
    print(objVal) 
    obj.append(objVal)
    UofClusters.append(U)  
    iterations.append(itr)
    VofClusters.append(V)
R = list()
for i in range(1,max(C) -2):
    r = abs( (obj[i] - obj[i+1]) / (obj[i-1] - obj[i]))   
    R.append(r)     
Cx = [3,4,5,6,7,8,9,10]
import matplotlib.pyplot as plt
index = findMin(R)
bestClusterNumber = np.array(index) 
Ubest = UofClusters[bestClusterNumber + 1]
Vbest = VofClusters[bestClusterNumber + 1]
sampleCluster = findCluster(Ubest)
plt.scatter(Z[:,0],Z[:,1],c=sampleCluster, cmap=plt.cm.Paired)

file = open("ClusterFile.txt","w+")
file.write(" x    y    Cluster\n" )
for i in range(train_size):
    for j in range(no_of_features):
            file.write(str("{:.2f}".format(Z[i][j])))
            file.write(" ")
    file.write("    ")
    file.write(str(sampleCluster[i]))
    file.write("\n")

file.close()

file = open("testfile.txt","w+")
for i in range(test_size):
    for j in range(no_of_features):
            file.write(str(Z_test[i][j]))
            file.write(" ")
    file.write("\n")
file.close()

file = open("centroidfile.txt","w+")
for i in range(bestClusterNumber + 3):
    for j in range(no_of_features):
            file.write(str("{:.4f}".format(Vbest[i][j])))
            file.write(" ")
    file.write("\n")
file.close()

fig,ax = plt.subplots()

ax.plot(C, obj,color="blue",marker="o")
ax.set_ylabel("Objective",color="blue",fontsize=14)

ax2=ax.twinx()
ax2.plot(C,iterations, color="red", marker="o")
ax2.set_xlabel("Cluster number",fontsize=14)
ax2.set_ylabel("Iterations",color="red",fontsize=14)

plt.show()
fig.savefig('Clusternumber vs Obj val vs Iter.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')

