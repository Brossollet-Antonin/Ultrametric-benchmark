# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:38:26 2019

@author: Antonin
"""

import random
from dataset import sort_dataset, sort_MNIST
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
    
    data = sort_MNIST(train=True)
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