# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 23:30:01 2020

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

corr = train_df.corr()
#    
#import pandas as pd
#import random
#import numpy as np 
#
#number_of_data_points = 10000                # Increase this to have a better model
#train_df_total = pd.read_csv('train_V2.csv')     # This might take a few seconds
#train_df = train_df_total.sample(n=number_of_data_points)    # Sampling 10,000 rows at random
#train_df.head()
#
#train_df = train_df.drop(columns = ['Id', 'groupId', 'matchId'])
#
#t = train_df.matchType.astype("category").cat.codes
#train_df = train_df.drop(columns = ['matchType'])
#ones = [1]*number_of_data_points
#train_df.insert(0,'Padding',ones)
#
#t = t.tolist()
#train_df['matchTpe'] = t
#l = (train_df.columns).tolist()
#l.pop()
#x = train_df[l].values.tolist()
#y = train_df['winPlacePerc'].values.tolist()
#
#sd = train_df.std(axis=0, skipna =True)
#sd = sd.tolist()
#m = train_df.mean(axis=0,skipna =True)
#m = m.tolist()
#for j in range(len(x[0])-1):
#    for i in range(number_of_data_points):
#        x[i][j+1] = (x[i][j+1]-m[j+1])/sd[j+1]
#
#x_train = []
#y_train = []
#x_test = []
#y_test = [] 
#train_size = 7000
#test_size = 3000
#
#for i in range(train_size):
#    x_train.append(x[i])
#    y_train.append(y[i])
#
#
#for i in range(train_size,number_of_data_points):
#    x_test.append(x[i])
#    y_test.append(y[i])
#
#
#def derv(oldTheta,j):
#    s=[] 
#    for i in range(train_size):
#        s1 = 0
#        for k in range(len(x_train[0])):
##            thetaT = np.array(oldTheta).T.tolist()
##           s = s +  ( np.dot(thetaT,x[i]) - y[i])*x[i]
#            s1 = s1 + oldTheta[k]*x_train[i][k]
#        s.append(s1 - y_train[i])
#    sum = 0
#    for i in range(train_size):
#        sum = sum + s[i]*x_train[i][j]
#    return sum/number_of_data_points
#
#
#noOfFeatures = train_df.shape[1]-1
#for i in range(test_size):
#    x_test[i][0] = 1
#
#for i in range(train_size):    
#    x_train[i][0] = 1
#
#theta = []
#for i in range(noOfFeatures):
#    theta.append(0)
#
## normalize, find corr, PCA
##corr = np.corrcoef(df)
#a = 0.01
#
#
#
#for i in range(40):
#    oldTheta = theta
#    for j in range(len(theta)):
#        theta[j] = theta[j] - a*derv(oldTheta,j)
##
#y_pred = []
#for i in range(test_size):
#    s = 0
#    for j in range(len(theta)):
#        s = s + theta[j]*x_test[i][j]
#    y_pred.append(s)
#
#acc = 0
#for i in range(test_size):
#    acc = acc + (y_test[i] - y_pred[i])**2
#print("Acc:",acc/test_size)
##number_of_test_points = 3000                # Increase this to have a better model
##test_df_total = pd.read_csv("train.csv")     # This might take a few seconds
##test_df = train_df_total.sample(n=number_of_test_points)    # Sampling 10,000 rows at random
##x_test = train_df[l].values.tolist()
##y_test = train_df['winPlacePerc'].values.tolist()
#
