# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:59:13 2019

@author: Antonin
"""

import numpy as np

M = np.array([[1,1,1,1],
              [1,1,0,0],
              [0,0,1,1],
              [1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]])

SVD = np.linalg.svd(M)
U=SVD[0][:,:4]
S=SVD[1]
V=SVD[2]

diagS=np.diag(S)
Rec=U@diagS@V

x1=np.array([[1],[0],[0],[0]])
x2=np.array([[0],[1],[0],[0]])
x3=np.array([[0],[0],[1],[0]])
x4=np.array([[0],[0],[0],[1]])
x=np.array([x1,x2,x3,x4])
y1=np.array([[1],[1],[0],[1],[0],[0],[0]])
y2=np.array([[1],[1],[0],[0],[1],[0],[0]])
y3=np.array([[1],[0],[1],[0],[0],[1],[0]])
y4=np.array([[1],[0],[1],[0],[0],[0],[1]])
y=np.array([y1,y2,y3,y4])

cor = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        cor[i,j]= np.transpose(y[i])@y[j]
    
svdcor=np.linalg.svd(cor)