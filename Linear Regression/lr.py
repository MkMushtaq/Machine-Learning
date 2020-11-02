# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:49:07 2020

@author: HP
"""

import pandas as pd
import numpy as np

number_of_data_points = 10000
train_df = pd.read_csv('train_V2.csv')
train_df = train_df.sample(n = number_of_data_points)

train_df = train_df.drop(columns = ['Id', 'groupId', 'matchId'])

t = train_df.matchType.astype("category").cat.codes

train_df = train_df.drop(columns = ['matchType'])

np.corrcoef(train_df)
ones = [1]*number_of_data_points
train_df.insert(0,'Padding',ones)

t = t.tolist()
train_df['matchTpe'] = t
l = (train_df.columns).tolist()
l.pop()
x = train_df[l].values.tolist()
y = train_df['winPlacePerc'].values.tolist()

sd = train_df.std(axis=0, skipna =True)
sd = sd.tolist()
minMax = (train_df.describe().loc[['min','max']].T).values.tolist()
m = train_df.mean(axis=0,skipna =True)
m = m.tolist()
for j in range(len(x[0])-1):
    for i in range(number_of_data_points):
        x[i][j+1] = (x[i][j+1]-m[j+1])/( minMax[j+1][1] - minMax[j+1][0] )
        
x_train = []
y_train = []
x_test = []
y_test = [] 
train_size = 7000
test_size = 3000

for i in range(train_size):
    x_train.append(x[i])
    y_train.append(y[i])


for i in range(train_size,number_of_data_points):
    x_test.append(x[i])
    y_test.append(y[i])
    
theta = []
noOfFeatures = len(x[0])
for i in range(noOfFeatures):
    theta.append(0)

learningRate = [0.5]


def derv(oldTheta,j):
    s = 0
    for k in range(train_size):
        h = 0
        for l in range(len(theta)):
            h = h + oldTheta[l]*x_train[k][l]
        s = s + (h - y_train[k])*x_train[k][j]
    return s/number_of_data_points

y_val = []
x_val = []
for a in learningRate:
    t = 0
    for iter in range(1,16):
        for i in range(iter):
            oldTheta = theta
            for j in range(len(theta)):
                theta[j] = theta[j] - a*derv(oldTheta,j)
                
        y_pred = []
        for i in range(test_size):
            s = 0
            for j in range(len(theta)):
                s = s + theta[j]*x_test[i][j]
            y_pred.append(s)
            
        rms = 0
        for i in range(test_size):
            rms = rms + abs(y_test[i] - y_pred[i])
        print("rms:",rms/test_size)
        y_val.append(rms/test_size)
        t = t + iter
        x_val.append(t)
        
    import matplotlib.pyplot as plt
    plt.plot(x_val,y_val)
    plt.show()
