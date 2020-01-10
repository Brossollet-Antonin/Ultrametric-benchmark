# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:38:26 2019

@author: Antonin
"""

import random
import sort_dataset
import numpy as np

def next_value(sequence, rates_matrix):
    i = sequence[-1]
    rates_vector = rates_matrix[i]
    lim = 0
    randomnum = random.random()
    for j,k in enumerate(rates_vector):
        lim += k 
        if randomnum <= lim:
            return j
    
    

def um_sequence_generator(sequence_first, rates_matrix, sequence_length, data):
    sequence = [sequence_first]
    print('Transition rates matrix :', rates_matrix)
    for i in range(sequence_length -1):
        sequence.append(next_value(sequence, rates_matrix))
    return sequence 

def um_sequence_generator_epoch(sequence_first, rates_matrix, epoch, data):
    sequence = [sequence_first]
    print('Transition rates matrix :', rates_matrix)
    compteur = np.array([0 for i in range(10)])
    size_labels_MNIST=np.array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
    while (compteur < epoch*size_labels_MNIST).any():
        next_value_seq = next_value(sequence, rates_matrix)
        sequence.append(next_value_seq)
        compteur[next_value_seq] +=1
    return sequence






def training_sequence(um_sequence):
    compteur = np.array([0 for i in range(10)])
    for k in um_sequence:
        compteur[k] += 1
    size_labels_MNIST=np.array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
    quotient = compteur//size_labels_MNIST
    nbr_clone = max(quotient) + 1
    print("Number of clones: ", nbr_clone)
    
    data = sort_dataset.sort_MNIST(train=True)
    iterable = [iter(data[i]) for i in range(len(data))] 
    train_sequence=[]
    for k in um_sequence:
        train_sequence.append(next(iterable[k]))
    return train_sequence
  
     


def sequence_autocor(um_sequence):
    length = len(um_sequence)
    autocor = []
    max_val = max(um_sequence)
    for dt in range(length):
        sumcor = 0
        for i in range(length - dt):
            if um_sequence[i] == um_sequence[i+dt]:
                sumcor += 1
            else:
                sumcor += -1/max_val
        autocor.append(sumcor/(length - dt))
    return autocor



#def training_sequence(um_sequence, batch_size, data):
#    iterable = [iter(data[i]) for i in range(len(data))] 
#    compteur = [0 for i in range(len(data))]
#    train_sequence = []
#    batch_cpt = 1
#    list_labels = []
#    for k in um_sequence:
#        compteur[k] += 1
#        if batch_cpt == 1:
#            samples, labels = next(iterable[k])
#            tensor_samples = samples
#            list_labels.append(labels)
#            
#        elif batch_cpt <= batch_size:
#            samples, labels = next(iterable[k])
#            tensor_samples = torch.cat((tensor_samples, samples), 0)
#            list_labels.append(labels)
#        if batch_cpt == batch_size:
#            batch_cpt = 0
#            tensor_labels = torch.tensor(list_labels)
#            next_batch = [tensor_samples, tensor_labels]
#            train_sequence.append(next_batch)
#            list_labels = []
#        if compteur[k] == 19:
#            compteur[k] = 0
#            random.shuffle(data[k])
#            iterable[k] = iter(data[k])
#        batch_cpt += 1
#    return train_sequence
#data = sort_dataset.sort_MNIST()
#
#
#um_sequence, rates_matrix= um_sequence_generator(0, 60000, data, 1/20, 10)
#import matplotlib.pyplot as plt
#um_autocor= sequence_autocor(um_sequence)
#plt.plot(um_autocor[:200])
#

#import copy
#
#shuffle_um_sequence = copy.deepcopy(um_sequence)
#random.shuffle(shuffle_um_sequence)
#autocor_control = sequence_autocor(shuffle_um_sequence)
#
#plt.figure()
#plt.plot(um_autocor[1:200], label='Utrametric')
#plt.title("Autocorrelation")
#plt.plot(autocor_control[1:200], label='Random')
#plt.legend()
#
#
#compteur=10*[0]
#for k in um_sequence:
#    compteur[k]+=1



#
#
#rates_matrix = rates_correlation.rates_cor(data, 1, 10)
#L=[sum(rates_matrix[i]) for i in range(10)]
#
#lim=0
#randomnum = random.random()
#for i,k in enumerate(rates_vector):
#    lim += k 
#    print('k',k)
#    print('lim',lim)
#    if randomnum <= lim:
#        print("True")
#next_value([0], rates)