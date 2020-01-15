# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:36:25 2019

@author: Antonin
"""

"""
Builds the rate transition vector from the avarage correlation between the different classes

"""

import torch
import numpy as np
from random import randint

def im_cor(A,B):
    avgA = torch.sum(A)/np.prod(A.size()) 
    avgB = torch.sum(B)/np.prod(B.size())
    return torch.sum((A-avgA)*(B-avgB))/(np.sqrt(torch.sum((A-avgA)**2))*np.sqrt(torch.sum((B-avgB)**2)))

def avg_cor(data, nbr_avg):
    length = len(data)
    cor_matrix = np.zeros((length, length))
    for i in range(length):
        for j in range(length):            
            temp_cor = 0
            for k in range(nbr_avg):
                indA, indB = [randint(0,len(data[i])-1) for a in range(10)], [randint(0,len(data[j])-1) for a in range(10)]
                for n in indA:
                    for m in indB: 
                        temp_cor += im_cor(data[i][n][0], data[j][m][0])
            cor_matrix[i][j] = temp_cor/(10*10*nbr_avg)     
    return cor_matrix
    
    

""" Attention ici on ne construit que les termes au dessus de la diagonale. Peut être qu'il sera nécessaire de construire la matrice complète
pour faciliter le reste du code. Mais il faut faire en sorte que les rates soient symétriques (artificiel d'avoir une asymétrie, ne serait pas
logique). Bonne façon de le coder serait en ajoutant la transposée moins la diagonale ? Ou sinon faire la somme de la matrice et sa transposée
et diviser par deux. 
En fait ne fontion pas car la renormalisation pour avoir une somme de lignes égale à 1 fait tout perdre. Mais c'est deux choses ne vont 
pas être compatibles... Ne peut pas à la fois forcer les lignes égales à 1 et la symétrie. Enfin si possible mais commence à devenir très très 
artificiel
"""

#def rates_cor(data, T, nbr_avg):
#    cor_matrix = avg_cor(data, nbr_avg)
#    rates_matrix = np.zeros(cor_matrix.shape)
#    for i in range(cor_matrix.shape[0]):
#        for j in range(cor_matrix.shape[1]):
#            rates_matrix[i][j] = np.exp(-(1-cor_matrix[i][j])/T)
#
#    rates_matrix = (rates_matrix + np.transpose(rates_matrix))/2
#    for i in range(cor_matrix.shape[0]):
#        sumrow = np.sum(rates_matrix[i])
#        rates_matrix[i] = rates_matrix[i]*(1/sumrow)    
#    return rates_matrix


def rates_cor(data, T, nbr_avg):
    cor_matrix = avg_cor(data, nbr_avg)
    rates_matrix = np.zeros(cor_matrix.shape)
    for i in range(cor_matrix.shape[0]):
        for j in range(cor_matrix.shape[1]):
            if i != j: 
                rates_matrix[i][j] = np.exp(-(1-cor_matrix[i][j])/T)
    rates_matrix = (rates_matrix + np.transpose(rates_matrix))/2
    for i in range(cor_matrix.shape[0]):
        sumrow = np.sum(rates_matrix[i])
        rates_matrix[i][i] = 1-sumrow   
        if rates_matrix[i][i] < 0:
            raise ValueError("Temperature too high, selfrates inferior to 0. Lower temperature")
    return rates_matrix







#
#data = sort_MNIST.sort_MNIST()
#
#cor = avg_cor(data, 10)
#
#T=1
#Rates=np.zeros((10,10))
#for i in range(10):
#    for j in range(10):
#        Rates[i][j] = np.exp(-(1-cor[i][j])/T)
#    
#
#    
#    
#
#
#
#A=torch.tensor(np.ones((10,10))*0.9)
#B=torch.tensor(np.ones((10,10))*0.9)
#im_cor(A,B)
#fullcor=[A for i in range(10)]
#fullmatrix=avg_cor(fullcor, 10)
#    
#
#imA=torch.tensor(np.identity(40))
#imB=torch.tensor(np.identity(40))
#for k in range(20):
#    imB[k][k]=0
#
#randA=torch.tensor(np.random.rand(40,40))
#randB=torch.tensor(np.random.rand(40,40))
#
#

