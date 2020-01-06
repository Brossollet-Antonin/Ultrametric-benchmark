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


path="C:/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Results/2019-07-16/ManualRates/MNIST/length10000000_batches10/T0.010 Memory0 block1 1563301163.430"
original_classes = np.load(path+"/var_original_accuracy.npy", 
        allow_pickle=True)
shuffle_classes = np.load(path+"/var_shuffle_accuracy.npy", 
        allow_pickle=True)

original_acc = np.load(path+"/var_original_accuracy.npy", allow_pickle=True)
shuffle_acc = np.load(path+"/var_shuffle_accuracy.npy", allow_pickle=True)
plt.figure()
plt.plot(original_acc[:,1], original_acc[:,0])
plt.title("original acc")
plt.figure()
plt.plot(shuffle_acc[:,1], shuffle_acc[:,0])
plt.title("shuffle acc")

original = pkl.load(open(path+"/original", 'rb'))
shuffle = pkl.load(open(path+"/shuffle", 'rb'))



original_class_prediction = np.load(path+'/var_original_classes_prediction.npy', allow_pickle=True)
shuffle_class_prediction = np.load(path+'/var_shuffle_classes_prediction.npy', allow_pickle=True)

fig, ax = plt.subplots(figsize=(9,9))
plt.plot(original)
plt.xlabel("Number of training examples", fontsize=22)
plt.ylabel("Label", fontsize=22)

for i in range(1,11):  
    plt.axvline(i*1e6, ls='--', c='r')




plt.figure()
plt.plot(shuffle)
plt.xlabel("Number of training examples", fontsize=22)
plt.ylabel("Label", fontsize=22)
ax.tick_params(labelsize=22)

x = np.arange(0,8)


nbr_row = 5
nbr_column = 2
fig, axs = plt.subplots(nbr_row, nbr_column, figsize=(9,9))


i,j = 0,0
for k in range(5,51,5):
    
    
    #fig, ax = plt.subplots(figsize=(9,9))
    barwidth=0.35

    axs[i,j].bar(x-barwidth/2,original_class_prediction[k,0][:8], width=barwidth, tick_label=[0,1,2,3,4,5,6,7])
    
    axs[i,j].bar(x+barwidth/2,shuffle_class_prediction[k,0][:8], width=barwidth)

    axs[i,j].set_title(shuffle_class_prediction[k,1], x=0.17, y=0.75, fontsize=16)
    axs[i,j].set_ylim([0,4000])
    axs[i,j].tick_params(labelsize=22)
    
    j+=1
    if j%nbr_column==0:
        i += 1
        j=0
     
axs[0,0].legend(['Original', 'Shuffle'], fontsize=12, loc='upper center')
        
for ax in axs.flat:
    ax.set_xlabel('Class predicted', fontsize=22)
    

for ax in axs.flat:
    ax.label_outer()



#train_data = pkl.load(open(path+'/train_data', 'rb'))
#data_shuffe = pkl.load(open(path+'/data_shuffle', 'rb'))
#
#
#seq_control_shuffle=[]
#for k in data_shuffe:
#    seq_control_shuffle.append(k[1].item())
#
#seq_control_original=[]
#for k in train_data:
#    seq_control_original.append(k[1].item())
#
##
#seq_control_block=[]
#for k in shuffledblock:
#    seq_control_block.append(k[1].item())
#
#
#
#
#
#










