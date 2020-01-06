# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 18:53:26 2019

@author: Antonin
"""

import numpy as np 
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


path= "C:/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Results/2019-07-02/Temperature/MNIST/length2000000_batches10/T0.130 Memory0 block1 1562112651.899/"

original_class_prediction = np.load(path+'var_original_classes_prediction.npy', allow_pickle=True)
shuffle_class_prediction = np.load(path+'var_shuffle_classes_prediction.npy', allow_pickle=True)

original_accuracy = np.load(path+'var_original_accuracy.npy', allow_pickle=True)
shuffle_accuracy = np.load(path+'var_shuffle_accuracy.npy', allow_pickle=True)

original = pkl.load(open(path+'original', 'rb'))
shuffle = pkl.load(open(path+'shuffle', 'rb'))

plt.figure()
plt.plot(original)
plt.figure()
plt.plot(shuffle)

x = np.arange(0,10)

for i in range(0,51,2):
    
    plt.figure()
#    plt.bar(x,original_class_prediction[i,0])
#    plt.title(original_class_prediction[i,1])
#    
    plt.bar(x,shuffle_class_prediction[i,0])
    plt.title(shuffle_class_prediction[i,1])

