# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:35:53 2019

@author: Antonin
"""

import numpy as np 
import matplotlib.pyplot as plt
import pickle as pkl 
import seaborn as sns

sns.set(style="ticks")


path="./Results/MNIST/length500000_batches10/T0.180 Memory0 block1 1562179360.813"
original_classes=np.load(path+"/var_original_accuracy.npy", 
        allow_pickle=True)

shuffle_classes=np.load(path+"/var_shuffle_accuracy.npy", 
        allow_pickle=True)

original = pkl.load(open(path+"/original", 'rb'))
shuffle = pkl.load(open(path+"/shuffle", 'rb'))



original_class_prediction = np.load(path+'/var_original_classes_prediction.npy', allow_pickle=True)
shuffle_class_prediction = np.load(path+'/var_shuffle_classes_prediction.npy', allow_pickle=True)

plt.figure()
plt.plot(original)
plt.title("Training sequence")
plt.xlabel("Number of training examples")
plt.figure()
plt.plot(shuffle)
plt.title("Shuffled sequence")
plt.xlabel("Number of training examples")

x = np.arange(0,10)

for i in range(0,5,1):
    
    plt.figure()
    plt.bar(x,original_class_prediction[i,0], alpha=0.8)
    
    plt.bar(x,shuffle_class_prediction[i,0], alpha=0.8)
    plt.title("Number of training samples: %s" % original_class_prediction[i,1])


    plt.legend(['Original', 'Shuffle'])
    plt.xlabel("Class predicted")
    