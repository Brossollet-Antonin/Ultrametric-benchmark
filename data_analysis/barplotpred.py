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


path="C:/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Results/2019-07-15/ManualRates/MNIST/length10000000_batches10/T0.010 Memory0 block1 1563285405.516"
original_classes=np.load(path+"/var_original_accuracy.npy", 
        allow_pickle=True)

shuffle_classes=np.load(path+"/var_shuffle_accuracy.npy", 
        allow_pickle=True)

original = pkl.load(open(path+"/original", 'rb'))
shuffle = pkl.load(open(path+"/shuffle", 'rb'))



original_class_prediction = np.load(path+'/var_original_classes_prediction.npy', allow_pickle=True)
shuffle_class_prediction = np.load(path+'/var_shuffle_classes_prediction.npy', allow_pickle=True)

fig, ax = plt.subplots(figsize=(9,9))
plt.plot(original)
plt.xlabel("Number of training examples", fontsize=22)
plt.ylabel("Label", fontsize=22)

plt.axvline(200000, ls='--', c='r')
plt.axvline(300000, ls='--', c='r')
plt.axvline(400000, ls='--', c='r')



plt.figure()
plt.plot(shuffle)
plt.xlabel("Number of training examples", fontsize=22)
plt.ylabel("Label", fontsize=22)
ax.tick_params(labelsize=22)

x = np.arange(0,10)

for i in range(0,6,1):

    fig, ax = plt.subplots(figsize=(9,9))
    barwidth=0.35
    plt.bar(x-barwidth/2,original_class_prediction[i,0], width=barwidth, tick_label=[0,1,2,3,4,5,6,7,8,9])
    
    plt.bar(x+barwidth/2,shuffle_class_prediction[i,0], width=barwidth)


    plt.legend(['Original', 'Shuffle'], fontsize=22)
    plt.xlabel("Class predicted", fontsize=22)
    ax.tick_params(labelsize=22)

    

    
    
train_data = pkl.load(open(path+'/train_data', 'rb'))
data_shuffe = pkl.load(open(path+'/data_shuffle', 'rb'))


seq_control_shuffle=[]
for k in data_shuffe:
    seq_control_shuffle.append(k[1].item())

seq_control_original=[]
for k in train_data:
    seq_control_original.append(k[1].item())

#
#seq_control_block=[]
#for k in shuffledblock:
#    seq_control_block.append(k[1].item())
#
#
#
#
#
#










