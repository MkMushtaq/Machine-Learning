# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:34:51 2020

@author: HP
"""
import pandas as pd
FAM = [[1,2,3,5,6,8,9],
       [1,1,2,3,4,6,8],
       [1,1,1,2,2,4,6],
       [1,1,1,1,2,3,5],
       [1,1,1,1,1,1,3],
       [1,1,1,1,1,1,2],
       [1,1,1,1,1,1,1]]
def f(x,a,b,c):
    if(x<=a):
        return 0
    elif(a<x<=b):
        return (x-a)/(b-a)      
    elif(b<x<c):                        
        return (c-x)/(c-b)
    else:
        return 0
                                      
                    
def g1(x,a,b,c):
    if(x<a):
        return 0
    elif(a<=x<b):
        return 1
    elif(b<=x<c):
        return (c-x)/(c-b)
    else:
        return 0
def g2(x,a,b,c):
    if(x<=a):
        return 0
    elif(a<x<=b):
        return (x-a)/(b-a)
    elif(b<x<c):
        return 1
    else:
        return 0

def centroid(x):
    if(x==1):
        return 25
    elif(x==2):
        return 50
    elif(x==3):
        return 60
    elif(x==4):
        return 70
    elif(x==5):
        return 75
    elif(x==6):
        return 80
    elif(x==7):
        return 85
    elif(x==8):
        return 90
    elif(x==9):
        return 95

    
def find_active_set(x):
    active_set = []
    active_set.append(g1(x,-1,-0.66,-0.33))
    active_set.append(f(x,-0.66,-0.33,0))
    active_set.append(f(x,-0.33,0,0.15))
    active_set.append(f(x,0,0.15,0.33))
    active_set.append(f(x,0.15,0.333,0.45))
    active_set.append(f(x,0.33,0.45,0.75))
    active_set.append(g2(x,0.45,0.75,1))
    
    return active_set
inp = pd.read_csv('input.csv') 
u = [10,30]
l = [-10,-30]

x1 = inp.loc[:,"x1"].tolist()
x2 = inp.loc[:,"x2"].tolist()
x = [x1,x2]
for j in range(0,len(u)):
    for i in range(0,len(x1)):
        x[j][i] = ( 2*x[j][i] - (u[j]+l[j]) )/(u[j]-l[j])
output = []

for i in range(0,len(x[0])):
    
        set1 = find_active_set(x[0][i])
        set2 = find_active_set(x[1][i])
        ind1 = [i for i in range(0,len(set1)) if set1[i]!=0]
        ind2 = [i for i in range(0,len(set2)) if set2[i]!=0]
        y = 0
        for i in ind1:
            for j in ind2:
                y = y + centroid(FAM[6-j][i])*set1[i]*set2[j]
        
        output.append(y)


y = output

col = ['x1','x2','Output']
data = pd.DataFrame(columns = col)
data['x1'] = inp.loc[:,"x1"].tolist()
data['x2'] = inp.loc[:,"x2"].tolist()
data['Output'] = y
print(data.head())
data.to_csv('17XJ1A0526.csv',index = False)
    