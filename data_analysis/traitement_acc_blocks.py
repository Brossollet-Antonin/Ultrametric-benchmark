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
from matplotlib.colors import LogNorm

sns.set(style="ticks")

temperature = ['0.150']
dataset='MNIST'
blocks = [1, 100, 1000, 10000, 100000]

totaldatalist = []

for T in temperature:
    for block in blocks:
    
    
        path = "C:/Users/Antonin/Documents/Documents/ENS 2A/Stage M1/Results/Blocks/"+dataset+"/NewShuffle/block"+str(block)+" T"+str(T)
        
        original_accuracy = np.load(path+'/0/var_original_accuracy.npy')
        
        for i in range(0,5):
            temp = np.load(path+'/'+str(i)+'/var_original_accuracy.npy')
            original_accuracy = np.concatenate((original_accuracy, temp), axis=0)
        
            
        size = np.shape(original_accuracy[:,1])
        modeo = np.zeros(size, dtype=object)
        modeo[:] = 'Original'
        original = pd.DataFrame(dict(nbr_ex=original_accuracy[:,1], acc=original_accuracy[:,0], 
                                     Mode=modeo, Block=(np.ones(size, dtype=int)*block)))
            
        shuffle_accuracy = np.load(path+'/0/var_shuffle_accuracy.npy')
        
        for i in range(0,5):
            temp = np.load(path+'/'+str(i)+'/var_shuffle_accuracy.npy')
            shuffle_accuracy = np.concatenate((shuffle_accuracy, temp), axis=0)
        
        size = np.shape(shuffle_accuracy[:,1])
        modes = np.zeros(size, dtype=object)
        modes[:] = 'Shuffle'   
            
        shuffle = pd.DataFrame(dict(nbr_ex=shuffle_accuracy[:,1], acc=shuffle_accuracy[:,0], Mode=modes, Block=(np.ones(size, dtype=int)*block)))
            
        concatenation = pd.concat([original,shuffle])
        
        fig, ax = plt.subplots(figsize=(9,9))
        intermediate_plot = sns.lineplot(x="nbr_ex", y='acc', data=concatenation, ci='sd', 
                                 style="Mode")
        ax.set_xlabel('Number of training samples', fontsize=15)
        ax.set_ylabel('Accuracy (%)', fontsize=15)
        intermediate_plot.set(ylim=(20,102))
        plt.legend(loc='best', fontsize=22)
        ax.tick_params(labelsize=22)
        sns.despine(top=False, bottom=False, right=False, left=False)

        totaldatalist += [concatenation]
    
totaldata = pd.concat(totaldatalist)


palette = sns.color_palette("GnBu_d", len(blocks))
palette.reverse()

fig, ax = plt.subplots(figsize=(9,9))
final_plot = sns.lineplot(x="nbr_ex", y='acc', data=totaldata, ci=None, 
                         hue="Block", hue_norm=LogNorm(), style="Mode", legend="full", palette=palette)
final_plot.set(ylim=(20,102))
sns.despine(top=False, bottom=False, right=False, left=False)
ax.set_xlabel('Number of training samples', fontsize=22)
ax.set_ylabel('Accuracy (%)', fontsize=22)
plt.legend(loc='lower right', fontsize=22, ncol=2)
ax.tick_params(labelsize=22)


