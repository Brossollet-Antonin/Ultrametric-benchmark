# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:59:41 2019

@author: Antonin
"""

import pdb
import numpy as np
from random import shuffle
from random import seed
import torch

def evaluate(net, dataset, device):
    # Return the accuracy, the predicted and real label for the whole test set and the difference between the two
    # Creation of the testing sequence 
    j = 0
    test_sequence=[]
    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data))]
    for i in range(len(dataset.test_data)):
        for j in range(dataset.class_sz_test):
            test_sequence.append(next(iterator[i]))
    shuffle(test_sequence)
    
    correct = 0
    total = 0
    # Array which will contain the predicted output, the ground truth and the difference of the two
    result = np.zeros((len(test_sequence), 3))
    
    with torch.no_grad():
        for i, data in enumerate(test_sequence):
            samples, labels = data
            samples, labels = samples.to(device), labels.to(device)
            outputs = net(samples)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            result[i][0] = predicted
            result[i][1] = labels
    result[:, 2] = np.abs(result[:, 0] - result[:, 1])
    return (100*correct/total, result)



def evaluate_hierarchical(trainer, device):
    # Return the accuracy, the predicted and GT and the distance between them for every hierachical level 
    # Creation of the testing sequence
    dataset = trainer.dataset
    if dataset.data_origin=='MNIST' or dataset.data_origin=='CIFAR10':
        excluded_labels = [8, 9]
    elif dataset.data_origin=='CIFAR100':
        excluded_labels = range(64, 100)
    
    else:
        excluded_labels = []    

    j = 0
    test_sequence=[]
    iterator = [iter(dataset.test_data[k]) for k in range(len(dataset.test_data) - len(excluded_labels))]   # Create the test sequence with only labels on which the network as been trained on 
    for i in range(len(dataset.test_data) - len(excluded_labels)):
        for j in range(dataset.class_sz_test):
            test_sequence.append(next(iterator[i]))
    
    shuffle(test_sequence)
    
    # Array which will contain the accuracies at the different hierarchical levels, 
    # the predicted output, the ground truth and the difference of the two
    result = [[], np.zeros((len(test_sequence), 3 + trainer.tree_depth))]
    with torch.no_grad():
        for i, data in enumerate(test_sequence):
            samples, labels = data
            if labels not in excluded_labels:
                samples, labels = samples.to(device), labels.to(device)
                outputs = trainer.network(samples)
                _, predicted = torch.max(outputs.data, 1)
                result[1][i][0] = predicted
                result[1][i][1] = labels
    zero = np.zeros(len(test_sequence))
    # Compute the difference between prediction and GT for every hierarchical level
    for i in range(2, trainer.tree_depth + 3):
        result[1][:, i] = np.abs((result[1][:, 0]//(trainer.tree_branching**(i-2))) 
                            - (result[1][:, 1]//(trainer.tree_branching**(i-2))))
        result[0].append((np.sum(result[1][:, i] == zero)/len(test_sequence))*100)
    
    return result


    

   