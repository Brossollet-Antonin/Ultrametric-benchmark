# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:34:57 2019

@author: Antonin
"""

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import pandas as pd

sns.set(style="ticks")

dataset="MNIST"
date="2019-07-10"
temperature = ['0.110', '0.130', '0.900']

totaldatalist = []

for T in temperature:
    
    
    path = "C:/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Results/Temperature/"+dataset+"/"+date+"/T" + T
    
    original_accuracy = np.load(path+'/0/var_original_accuracy.npy')
    
    for i in range(1,10):
        temp = np.load(path+'/'+str(i)+'/var_original_accuracy.npy')
        original_accuracy = np.concatenate((original_accuracy, temp), axis=0)
    
        
    size = np.shape(original_accuracy[:,1])
    modeo = np.zeros(size, dtype=object)
    modeo[:] = 'Original'
    original = pd.DataFrame(dict(nbr_ex=original_accuracy[:,1], 
                                 acc=original_accuracy[:,0], Mode=modeo, 
                                 Temperature=np.ones(size)*float(T)))
        
    shuffle_accuracy = np.load(path+'/0/var_shuffle_accuracy.npy')

    
    for i in range(1,10):
        temp = np.load(path+'/'+str(i)+'/var_shuffle_accuracy.npy')
        shuffle_accuracy = np.concatenate((shuffle_accuracy, temp), axis=0)
    
    size = np.shape(shuffle_accuracy[:,1])
    modes = np.zeros(size, dtype=object)
    modes[:] = 'Shuffle'   
        
    shuffle = pd.DataFrame(dict(nbr_ex=shuffle_accuracy[:,1], 
                                acc=shuffle_accuracy[:,0], Mode=modes,
                                Temperature=np.ones(size)*float(T)))
        
    concatenation = pd.concat([original,shuffle])

    fig, ax = plt.subplots(figsize=(9,9))

    intermediate_plot = sns.lineplot(x="nbr_ex", y='acc', data=concatenation, ci='sd', 
                             style="Mode")

    plt.legend(loc='best', fontsize=22)
    ax.tick_params(labelsize=22)
    #plt.title("Accuracy Temperature="+str(T))
    #intermediate_plot.set_axis_labels("Number of training samples", "Accuracy (%)")
    intermediate_plot.set(ylim=(20,102))
    sns.despine(top=False, bottom=False, right=False, left=False)
    ax.set_xlabel('Number of training samples', fontsize=22)
    ax.set_ylabel('Accuracy (%)', fontsize=22)
    
    totaldatalist += [concatenation]
    

totaldata = pd.concat(totaldatalist)


palette = sns.color_palette("GnBu_d", len(temperature))


fig, ax = plt.subplots(figsize=(9,9))

final_plot = sns.lineplot(x="nbr_ex", y='acc', data=totaldata, ci=None, 
                         hue="Temperature", style="Mode", palette=palette)
plt.legend(loc='best', fontsize=22, ncol=2)
ax.tick_params(labelsize=22)
final_plot.set(ylim=(20,102))


sns.despine(top=False, bottom=False, right=False, left=False)
ax.set_xlabel('Number of training samples', fontsize=22)
ax.set_ylabel('Accuracy (%)', fontsize=22)

#
#original_seq5 = pkl.load(open("C:/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Results/Temperature/"+dataset+"/"+date+"/T" +'0.150/' +'2/original', 'rb'))
#shuffle_seq5 = pkl.load(open("C:/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Results/Temperature/"+dataset+"/"+date+"/T" +'0.150/' +'2/shuffle', 'rb'))
#
#original_seq7 = pkl.load(open("C:/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Results/Temperature/"+dataset+"/"+date+"/T" +'0.170/' +'2/original', 'rb'))
#shuffle_seq7 = pkl.load(open("C:/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Results/Temperature/"+dataset+"/"+date+"/T" +'0.170/' +'2/shuffle', 'rb'))
#
#original_seq9 = pkl.load(open("C:/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Results/Temperature/"+dataset+"/"+date+"/T" +'0.190/' +'2/original', 'rb'))
#shuffle_seq9 = pkl.load(open("C:/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Results/Temperature/"+dataset+"/"+date+"/T" +'0.190/' +'2/shuffle', 'rb'))    
#    
#fig, ax = plt.subplots(figsize=(9,9))
#plt.plot(original_seq5)
#ax.tick_params(labelsize=22)
#ax.set_xlabel('Number of training samples', fontsize=15)
#ax.set_ylabel('Label', fontsize=15)
#fig, ax = plt.subplots(figsize=(9,9))
#plt.plot(original_seq7)
#ax.tick_params(labelsize=22)
#ax.set_xlabel('Number of training samples', fontsize=15)
#ax.set_ylabel('Label', fontsize=15)
#fig, ax = plt.subplots(figsize=(9,9))
#plt.plot(original_seq9)
#ax.tick_params(labelsize=22)
#ax.set_xlabel('Number of training samples', fontsize=15)
#ax.set_ylabel('Label', fontsize=15)
#    
